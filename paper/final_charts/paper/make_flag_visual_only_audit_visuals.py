#!/usr/bin/env python3
"""Make paper-facing visuals for the visual-only 4o vs 5.4 crop audit.

The intended source is a paired single-agent crop-audit CSV in which both
models answer the same image crop without social-evidence content.
The script is deliberately schema-tolerant so pilot files can use either
explicit audit columns or the columns emitted by the existing flag probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memetic-drift-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memetic-drift-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_CSV = (
    ROOT / "results" / "flag_game" / "visual_only_paired_crop_audit" / "results.csv"
)
DEFAULT_FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DEFAULT_DATA_DIR = ROOT / "paper" / "exports" / "data"
DEFAULT_STEM = "flag_game_visual_only_paired_audit"

DEFAULT_LEFT_MODEL = "gpt-4o"
DEFAULT_RIGHT_MODEL = "gpt-5.4"
KNOWN_MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-5.4": "GPT-5.4",
}

INFO_ORDER = ["unique", "narrow", "moderate", "ambiguous", "unknown"]
INFO_LABELS = {
    "unique": "Unique",
    "narrow": "Narrow",
    "moderate": "Moderate",
    "ambiguous": "Ambiguous",
    "unknown": "Unknown",
}
INFO_ALIASES = {
    "single": "unique",
    "singleton": "unique",
    "unique_crop": "unique",
    "low": "narrow",
    "weak": "narrow",
    "medium": "moderate",
    "mid": "moderate",
    "high": "ambiguous",
    "uninformative": "ambiguous",
}

MODEL_COLORS = {
    DEFAULT_LEFT_MODEL: HOUSE_COLORS["blue"],
    DEFAULT_RIGHT_MODEL: HOUSE_COLORS["red"],
}
PAPER_SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]

PAIR_CONTEXT_COLUMNS = [
    "protocol",
    "prompt_variant",
    "m",
    "rep",
    "seed",
    "crop_condition",
    "false_memory_count",
    "true_memory_count",
    "lure_country",
    "lure_relation",
]
PAIR_ID_COLUMNS = ["pair_id", "stimulus_id", "crop_id", "crop_path"]
AUTO_PAIR_COLUMNS = [
    "truth_country",
    "image_country",
    "stimulus_country",
    "image_id",
    "crop_index",
    "crop_crop_index",
    "crop_id",
    "crop_path",
    "crop_left",
    "crop_top",
    "crop_width",
    "crop_height",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
    *PAIR_CONTEXT_COLUMNS,
]


def add_paper_sans_font() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return


def apply_paper_sans_rc() -> None:
    add_paper_sans_font()
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_SANS_FONTS,
            "mathtext.fontset": "dejavusans",
            "font.size": 8.3,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visual-only paired crop-audit figures comparing GPT-4o "
            "and GPT-5.4."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=f"Path to the paired visual-only audit CSV. Default: {DEFAULT_SOURCE_CSV}",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Directory for PNG/PDF/SVG outputs. Default: {DEFAULT_FIGURE_DIR}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for summary CSV/JSON outputs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"Output filename stem. Default: {DEFAULT_STEM}",
    )
    parser.add_argument(
        "--left-model",
        default=DEFAULT_LEFT_MODEL,
        help=f"Model used as the first paired baseline. Default: {DEFAULT_LEFT_MODEL}",
    )
    parser.add_argument(
        "--right-model",
        default=DEFAULT_RIGHT_MODEL,
        help=f"Model used as the second paired baseline. Default: {DEFAULT_RIGHT_MODEL}",
    )
    parser.add_argument(
        "--pair-columns",
        default=None,
        help=(
            "Comma-separated columns defining matched prompts/crops. If omitted, "
            "the script infers a crop/stimulus key."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=12,
        help=(
            "Number of GPT-4o-compatible/GPT-5.4-incompatible example candidates "
            "to export to the gallery CSV/figure. Default: 12."
        ),
    )
    parser.add_argument(
        "--selected-example-pair-id",
        action="append",
        dest="selected_example_pair_ids",
        default=None,
        help=(
            "Pair ID to force into the three-example qualitative figure. Repeat "
            "to set the figure order. If omitted, the script uses the top three "
            "ranked GPT-4o-compatible/GPT-5.4-incompatible examples."
        ),
    )
    parser.add_argument(
        "--manual-adjudication",
        type=Path,
        default=None,
        help=(
            "Optional filled adjudication CSV. Use manual_<model_slug>_label "
            "columns with values correct, compatible, or wrong to override the "
            "automatic crop-cache categories in the main panel."
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing visual-only audit CSV: {path}\n"
            "Expected one row per model response with at least model, "
            "truth_country, and choice_country/predicted_country. The default "
            "location is reserved for the paired crop-audit run."
        )

    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def first_value(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "t", "yes", "y", "correct"}:
        return True
    if text in {"0", "false", "f", "no", "n", "wrong", "incorrect"}:
        return False
    return default


def normalize_country(value: Any) -> str:
    return str(value or "").strip()


def country_key(value: Any) -> str:
    return normalize_country(value).casefold()


def parse_country_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [normalize_country(item) for item in parsed if normalize_country(item)]

    for separator in ("|", ";", ","):
        if separator in text:
            return [
                normalize_country(part)
                for part in text.split(separator)
                if normalize_country(part)
            ]
    return [text]


def parse_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def infer_informativeness_label(row: dict[str, Any], compatible_count: int | None) -> str:
    raw_label = first_value(
        row,
        [
            "crop_informativeness_label",
            "informativeness_label",
            "crop_info_label",
            "crop_condition",
        ],
    )
    if raw_label is not None:
        label = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
        label = INFO_ALIASES.get(label, label)
        if label in INFO_ORDER:
            return label

    if compatible_count is None:
        return "unknown"
    if compatible_count <= 1:
        return "unique"
    if compatible_count <= 4:
        return "narrow"
    if compatible_count <= 8:
        return "moderate"
    return "ambiguous"


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        model = normalize_country(first_value(row, ["model", "model_name"]))
        truth = normalize_country(
            first_value(row, ["truth_country", "target_country", "true_country"])
        )
        choice = normalize_country(
            first_value(
                row,
                [
                    "choice_country",
                    "predicted_country",
                    "prediction_country",
                    "final_country",
                    "country",
                    "answer_country",
                ],
            )
        )
        valid = parse_bool(first_value(row, ["valid", "is_valid"]), default=True)

        correct_value = first_value(
            row, ["correct", "is_correct", "exact_correct", "truth_correct"]
        )
        exact_correct = (
            parse_bool(correct_value)
            if correct_value is not None
            else valid and truth and country_key(choice) == country_key(truth)
        )

        compatible_countries = parse_country_list(
            first_value(
                row,
                [
                    "crop_compatible_countries",
                    "compatible_countries",
                    "compatible_country_list",
                ],
            )
        )
        compatible_count = parse_int(
            first_value(
                row,
                [
                    "crop_compatible_country_count",
                    "compatible_country_count",
                    "num_compatible_countries",
                ],
            )
        )
        if compatible_count is None and compatible_countries:
            compatible_count = len(compatible_countries)

        compatible_value = first_value(
            row,
            [
                "choice_crop_compatible",
                "crop_compatible",
                "compatible_with_crop",
                "choice_compatible",
            ],
        )
        if compatible_value is not None:
            choice_crop_compatible = parse_bool(compatible_value)
        elif compatible_countries:
            compatible_keys = {country_key(country) for country in compatible_countries}
            choice_crop_compatible = valid and country_key(choice) in compatible_keys
        else:
            choice_crop_compatible = exact_correct

        info_label = infer_informativeness_label(row, compatible_count)

        out: dict[str, Any] = dict(row)
        out.update(
            {
                "_row_index": index,
                "model": model,
                "truth_country": truth,
                "choice_country": choice,
                "valid": valid,
                "exact_correct": bool(valid and exact_correct),
                "choice_crop_compatible": bool(valid and choice_crop_compatible),
                "crop_compatible_country_count": compatible_count,
                "crop_informativeness_label": info_label,
            }
        )
        normalized.append(out)
    return normalized


def infer_pair_columns(
    rows: list[dict[str, Any]], override: str | None
) -> list[str]:
    if override:
        columns = [column.strip() for column in override.split(",") if column.strip()]
        missing = [column for column in columns if all(column not in row for row in rows)]
        if missing:
            raise ValueError(f"Pair column(s) not found in source CSV: {', '.join(missing)}")
        return columns

    present_columns = {column for row in rows for column in row.keys()}
    for id_column in PAIR_ID_COLUMNS:
        if id_column in present_columns:
            columns = [id_column]
            columns.extend(
                column
                for column in ["protocol", "prompt_variant", "m", "rep"]
                if column in present_columns and column != id_column
            )
            return columns

    columns = [column for column in AUTO_PAIR_COLUMNS if column in present_columns]
    if not columns:
        raise ValueError(
            "Could not infer paired crop columns. Pass --pair-columns with a "
            "comma-separated key such as pair_id or truth_country,crop_index,seed,rep."
        )
    return columns


def pair_key(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in columns)


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    pair_columns: list[str],
    left_model: str,
    right_model: str,
) -> tuple[list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]], dict[str, int]]:
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["model"] not in {left_model, right_model}:
            continue
        grouped[pair_key(row, pair_columns)][row["model"]].append(row)

    pairs: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]] = []
    stats = {
        "candidate_keys": len(grouped),
        "paired_keys": 0,
        "unpaired_left_only_keys": 0,
        "unpaired_right_only_keys": 0,
        "duplicate_model_key_entries": 0,
    }
    for key, by_model in grouped.items():
        left_rows = by_model.get(left_model, [])
        right_rows = by_model.get(right_model, [])
        if not left_rows and right_rows:
            stats["unpaired_right_only_keys"] += 1
            continue
        if left_rows and not right_rows:
            stats["unpaired_left_only_keys"] += 1
            continue
        stats["paired_keys"] += 1
        if len(left_rows) != 1 or len(right_rows) != 1:
            stats["duplicate_model_key_entries"] += (
                max(0, len(left_rows) - 1) + max(0, len(right_rows) - 1)
            )
        for index in range(min(len(left_rows), len(right_rows))):
            pairs.append((key, left_rows[index], right_rows[index]))
    return pairs, stats


def mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    array = np.array(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(len(array)))


def summarize_model_correctness(
    rows: list[dict[str, Any]], *, models: list[str]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model in models:
        cell = [row for row in rows if row["model"] == model]
        correct_values = [float(row["exact_correct"]) for row in cell]
        accuracy_mean, accuracy_sem = mean_sem(correct_values)
        correct_count = int(sum(correct_values))
        summary.append(
            {
                "model": model,
                "n": len(cell),
                "correct_count": correct_count,
                "incorrect_count": len(cell) - correct_count,
                "accuracy_mean": accuracy_mean,
                "accuracy_sem": accuracy_sem,
                "valid_count": sum(1 for row in cell if bool(row.get("valid"))),
            }
        )
    return summary


def summarize_model_crop_compatibility(
    rows: list[dict[str, Any]], *, models: list[str]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model in models:
        cell = [row for row in rows if row["model"] == model]
        compatible_values = [float(row["choice_crop_compatible"]) for row in cell]
        compatibility_mean, compatibility_sem = mean_sem(compatible_values)
        compatible_count = int(sum(compatible_values))
        summary.append(
            {
                "model": model,
                "n": len(cell),
                "compatible_count": compatible_count,
                "incompatible_count": len(cell) - compatible_count,
                "compatibility_mean": compatibility_mean,
                "compatibility_sem": compatibility_sem,
                "valid_count": sum(1 for row in cell if bool(row.get("valid"))),
            }
        )
    return summary


MANUAL_LABEL_ALIASES = {
    "correct": "correct",
    "true": "correct",
    "exact": "correct",
    "exact_correct": "correct",
    "compatible": "compatible",
    "plausible": "compatible",
    "visually_compatible": "compatible",
    "wrong": "wrong",
    "incorrect": "wrong",
    "incompatible": "wrong",
    "off_cache": "wrong",
    "off-cache": "wrong",
}


def model_slug(model: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in model)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def auto_visual_label(row: dict[str, Any]) -> str:
    if bool(row["exact_correct"]):
        return "correct"
    if bool(row["choice_crop_compatible"]):
        return "compatible"
    return "off_cache"


def normalize_manual_label(value: Any, *, path: Path | None = None, row_number: int | None = None) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text:
        return None
    label = MANUAL_LABEL_ALIASES.get(text)
    if label is None:
        location = ""
        if path is not None and row_number is not None:
            location = f" at {path}:{row_number}"
        raise ValueError(
            f"Unrecognized manual label{location}: {value!r}. "
            "Use correct, compatible, or wrong."
        )
    return label


def read_manual_adjudication(path: Path, *, models: list[str]) -> dict[tuple[str, str], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manual adjudication CSV: {path}")
    labels: dict[tuple[str, str], str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            pair_id = str(row.get("pair_id") or "").strip()
            if not pair_id:
                continue
            long_model = str(row.get("model") or "").strip()
            if long_model:
                manual_value = first_value(
                    row,
                    [
                        "manual_label",
                        "manual_visual_label",
                        "visual_label",
                        "adjudicated_label",
                        "label",
                    ],
                )
                label = normalize_manual_label(
                    manual_value,
                    path=path,
                    row_number=row_number,
                )
                if label is None and "choice_crop_compatible" in row:
                    if parse_bool(row.get("exact_correct"), default=False):
                        label = "correct"
                    elif parse_bool(row.get("choice_crop_compatible"), default=False):
                        label = "compatible"
                    else:
                        label = "wrong"
                if label is not None:
                    labels[(pair_id, long_model)] = label
                continue

            for model in models:
                slug = model_slug(model)
                value = first_value(
                    row,
                    [
                        f"manual_{slug}_label",
                        f"{slug}_manual_label",
                        f"manual_{model}_label",
                    ],
                )
                label = normalize_manual_label(value, path=path, row_number=row_number)
                if label is not None:
                    labels[(pair_id, model)] = label
    return labels


def summarize_model_visual_read_categories(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    manual_labels: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    has_manual_labels = bool(manual_labels)
    for model in models:
        cell = [row for row in rows if row["model"] == model]
        correct_count = 0
        compatible_only_count = 0
        third_count = 0
        manual_labeled_count = 0
        for row in cell:
            pair_id = str(row.get("pair_id") or "")
            label = (manual_labels or {}).get((pair_id, model))
            if label is not None:
                manual_labeled_count += 1
            else:
                label = auto_visual_label(row)

            if label == "correct":
                correct_count += 1
            elif label == "compatible":
                compatible_only_count += 1
            else:
                third_count += 1

        n = len(cell)
        denominator = n if n else 1
        third_label = "Wrong" if has_manual_labels else "Off-cache"
        summary.append(
            {
                "model": model,
                "n": n,
                "correct_count": correct_count,
                "compatible_count": compatible_only_count,
                "third_count": third_count,
                "correct_share": correct_count / denominator,
                "compatible_share": compatible_only_count / denominator,
                "third_share": third_count / denominator,
                "third_label": third_label,
                "manual_labeled_count": manual_labeled_count,
                "valid_count": sum(1 for row in cell if bool(row.get("valid"))),
            }
        )
    return summary


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def summarize_pairs(
    pairs: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    b = 0
    c = 0
    both_correct = 0
    both_wrong = 0
    for _, left, right in pairs:
        left_correct = bool(left["exact_correct"])
        right_correct = bool(right["exact_correct"])
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct and not right_correct:
            b += 1
        elif right_correct and not left_correct:
            c += 1
        else:
            both_wrong += 1
    return {
        "n_pairs": len(pairs),
        "left_correct_right_wrong": b,
        "right_correct_left_wrong": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_p(b, c),
    }


def summarize_pair_crop_compatibility(
    pairs: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    left_only = 0
    right_only = 0
    both_compatible = 0
    both_incompatible = 0
    for _, left, right in pairs:
        left_compatible = bool(left["choice_crop_compatible"])
        right_compatible = bool(right["choice_crop_compatible"])
        if left_compatible and right_compatible:
            both_compatible += 1
        elif left_compatible and not right_compatible:
            left_only += 1
        elif right_compatible and not left_compatible:
            right_only += 1
        else:
            both_incompatible += 1
    return {
        "n_pairs": len(pairs),
        "left_compatible_right_incompatible": left_only,
        "right_compatible_left_incompatible": right_only,
        "both_compatible": both_compatible,
        "both_incompatible": both_incompatible,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def paired_outcome_rows(
    pair_summary: dict[str, Any], *, models: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "outcome": "both_correct",
            "label": "Both correct",
            "count": int(pair_summary["both_correct"]),
        },
        {
            "outcome": "left_only_correct",
            "label": f"{display_model(models[0])} only",
            "count": int(pair_summary["left_correct_right_wrong"]),
        },
        {
            "outcome": "right_only_correct",
            "label": f"{display_model(models[1])} only",
            "count": int(pair_summary["right_correct_left_wrong"]),
        },
        {
            "outcome": "both_wrong",
            "label": "Both wrong",
            "count": int(pair_summary["both_wrong"]),
        },
    ]


def paired_compatibility_outcome_rows(
    pair_summary: dict[str, Any], *, models: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "outcome": "both_compatible",
            "label": "Both compatible",
            "count": int(pair_summary["both_compatible"]),
        },
        {
            "outcome": "left_only_compatible",
            "label": f"{display_model(models[0])} only",
            "count": int(pair_summary["left_compatible_right_incompatible"]),
        },
        {
            "outcome": "right_only_compatible",
            "label": f"{display_model(models[1])} only",
            "count": int(pair_summary["right_compatible_left_incompatible"]),
        },
        {
            "outcome": "both_incompatible",
            "label": "Both incompatible",
            "count": int(pair_summary["both_incompatible"]),
        },
    ]


def resolve_crop_path(value: Any, *, source: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        if "results" in path.parts:
            rel = Path(*path.parts[path.parts.index("results") :])
            bundled = ROOT / rel
            if bundled.exists():
                return bundled
        return path
    candidate = source.parent / path
    if candidate.exists():
        return candidate
    return ROOT / path


def display_country_list(value: Any) -> str:
    return " | ".join(parse_country_list(value))


def paired_manual_adjudication_rows(
    pairs: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]],
    *,
    models: list[str],
    source: Path,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for key, left, right in pairs:
        crop_path = resolve_crop_path(first_value(left, ["crop_path", "image_path"]), source=source)
        if crop_path is None:
            crop_path = resolve_crop_path(
                first_value(right, ["crop_path", "image_path"]),
                source=source,
            )
        pair_id = (
            first_value(left, ["pair_id", "stimulus_id", "crop_id"])
            or first_value(right, ["pair_id", "stimulus_id", "crop_id"])
            or "|".join(key)
        )
        label = str(left.get("crop_informativeness_label") or right.get("crop_informativeness_label") or "unknown")
        output: dict[str, Any] = {
            "pair_id": pair_id,
            "pair_key": "|".join(key),
            "truth_country": left.get("truth_country") or right.get("truth_country"),
            "crop_path": str(crop_path or ""),
            "crop_informativeness_label": label,
            "crop_compatible_country_count": (
                left.get("crop_compatible_country_count")
                or right.get("crop_compatible_country_count")
            ),
            "crop_compatible_countries": display_country_list(
                first_value(
                    left,
                    ["crop_compatible_countries", "compatible_countries"],
                )
                or first_value(
                    right,
                    ["crop_compatible_countries", "compatible_countries"],
                )
            ),
            "manual_notes": "",
        }
        for model, row in [(models[0], left), (models[1], right)]:
            slug = model_slug(model)
            output[f"{slug}_choice_country"] = row.get("choice_country")
            output[f"{slug}_reason"] = first_value(row, ["reason", "clue"])
            output[f"{slug}_exact_correct"] = row.get("exact_correct")
            output[f"{slug}_cache_crop_compatible"] = row.get("choice_crop_compatible")
            output[f"{slug}_auto_label"] = auto_visual_label(row)
            output[f"manual_{slug}_label"] = ""
            output[f"manual_{slug}_notes"] = ""
        output_rows.append(output)
    output_rows.sort(key=lambda row: str(row.get("pair_id")))
    return output_rows


def select_example_pairs(
    pairs: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]],
    *,
    source: Path,
    max_examples: int | None = 3,
) -> list[dict[str, Any]]:
    info_rank = {label: index for index, label in enumerate(INFO_ORDER)}
    candidates: list[dict[str, Any]] = []
    for key, left, right in pairs:
        if not bool(left["choice_crop_compatible"]) or bool(right["choice_crop_compatible"]):
            continue
        crop_path = resolve_crop_path(first_value(left, ["crop_path", "image_path"]), source=source)
        if crop_path is None or not crop_path.exists():
            crop_path = resolve_crop_path(
                first_value(right, ["crop_path", "image_path"]),
                source=source,
            )
        if crop_path is None or not crop_path.exists():
            continue

        label = str(left.get("crop_informativeness_label") or "unknown")
        pair_id = (
            first_value(left, ["pair_id", "stimulus_id", "crop_id"])
            or first_value(right, ["pair_id", "stimulus_id", "crop_id"])
            or "|".join(key)
        )
        candidates.append(
            {
                "pair_key": "|".join(key),
                "pair_id": pair_id,
                "crop_path": str(crop_path),
                "truth_country": left.get("truth_country") or right.get("truth_country"),
                "crop_informativeness_label": label,
                "crop_compatible_country_count": left.get("crop_compatible_country_count"),
                "left_model": left.get("model"),
                "left_choice_country": left.get("choice_country"),
                "left_reason": first_value(left, ["reason", "clue"]),
                "left_crop_compatible": left.get("choice_crop_compatible"),
                "right_model": right.get("model"),
                "right_choice_country": right.get("choice_country"),
                "right_reason": first_value(right, ["reason", "clue"]),
                "right_crop_compatible": right.get("choice_crop_compatible"),
                "left_exact_correct": left.get("exact_correct"),
                "right_exact_correct": right.get("exact_correct"),
                "failure_category": "right_crop_incompatible",
                "_rank": (
                    0 if bool(left.get("exact_correct")) else 1,
                    info_rank.get(label, len(INFO_ORDER)),
                    str(left.get("truth_country")),
                    str(pair_id),
                ),
            }
        )
    candidates.sort(key=lambda row: row["_rank"])
    for rank, row in enumerate(candidates, start=1):
        row["failure_rank"] = rank
        row["paper_default_example"] = False
        row.pop("_rank", None)
    if max_examples is None:
        return candidates
    return candidates[:max_examples]


def parse_selected_pair_ids(raw_values: list[str] | None) -> list[str]:
    pair_ids: list[str] = []
    for raw_value in raw_values or []:
        for part in str(raw_value).split(","):
            pair_id = part.strip()
            if pair_id:
                pair_ids.append(pair_id)
    return pair_ids


def selected_examples_by_pair_ids(
    candidates: list[dict[str, Any]], pair_ids: list[str]
) -> list[dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in candidates:
        lookup.setdefault(str(row.get("pair_id")), row)

    missing = [pair_id for pair_id in pair_ids if pair_id not in lookup]
    if missing:
        raise ValueError(
            "Requested --selected-example-pair-id value(s) were not found among "
            "GPT-4o-compatible/GPT-5.4-incompatible candidates with existing crop_path: "
            f"{', '.join(missing)}"
        )

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair_id in pair_ids:
        if pair_id in seen:
            continue
        seen.add(pair_id)
        row = dict(lookup[pair_id])
        row["paper_default_example"] = True
        selected.append(row)
    return selected


def display_model(model: str) -> str:
    return KNOWN_MODEL_LABELS.get(model, model)


def model_color(model: str, fallback: str) -> str:
    return MODEL_COLORS.get(model, fallback)


def finite_or_zero(value: float) -> float:
    return 0.0 if math.isnan(value) else value


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )


def draw_crop_compatibility_panel(
    ax: plt.Axes,
    model_crop_compatibility: list[dict[str, Any]],
    *,
    models: list[str],
) -> None:
    lookup = {row["model"]: row for row in model_crop_compatibility}
    x = np.arange(len(models), dtype=float)
    compatible_rates = [
        finite_or_zero(float(lookup.get(model, {}).get("compatibility_mean", math.nan)))
        for model in models
    ]
    incompatible_rates = [1.0 - value for value in compatible_rates]
    ax.bar(
        x,
        compatible_rates,
        width=0.58,
        color=HOUSE_COLORS["teal"],
        edgecolor=HOUSE_COLORS["black"],
        linewidth=0.8,
        label="Crop-compatible",
    )
    ax.bar(
        x,
        incompatible_rates,
        width=0.58,
        bottom=compatible_rates,
        color=HOUSE_COLORS["light_gray"],
        edgecolor=HOUSE_COLORS["black"],
        linewidth=0.8,
        label="Crop-incompatible",
    )
    for index, model in enumerate(models):
        row = lookup.get(model, {})
        compatible_count = int(row.get("compatible_count", 0) or 0)
        incompatible_count = int(row.get("incompatible_count", 0) or 0)
        n = int(row.get("n", 0) or 0)
        if compatible_rates[index] > 0.08:
            ax.text(
                x[index],
                compatible_rates[index] / 2,
                f"{compatible_count} compatible",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        if incompatible_rates[index] > 0.08:
            ax.text(
                x[index],
                compatible_rates[index] + incompatible_rates[index] / 2,
                f"{incompatible_count} incompatible",
                ha="center",
                va="center",
                fontsize=8,
                color=HOUSE_COLORS["black"],
            )
        ax.text(
            x[index],
            1.035,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=HOUSE_COLORS["gray"],
        )
    ax.set_title("Crop-compatible answer or not")
    ax.set_ylabel("Share of crop prompts")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in models])
    ax.set_ylim(0, 1.12)
    style_axis(ax, grid=True)


def draw_visual_read_category_panel(
    ax: plt.Axes,
    model_visual_read_categories: list[dict[str, Any]],
    *,
    models: list[str],
) -> None:
    lookup = {row["model"]: row for row in model_visual_read_categories}
    categories = [
        ("correct", "Correct", "correct_count", "correct_share", HOUSE_COLORS["teal"]),
        (
            "compatible",
            "Compatible",
            "compatible_count",
            "compatible_share",
            HOUSE_COLORS["gold"],
        ),
        (
            "third",
            str(model_visual_read_categories[0].get("third_label", "Off-cache")),
            "third_count",
            "third_share",
            HOUSE_COLORS["light_gray"],
        ),
    ]

    x = np.arange(len(models), dtype=float) * 0.72
    bottoms = np.zeros(len(models), dtype=float)
    for category_index, (_, label, count_key, share_key, color) in enumerate(categories):
        counts = [
            float(lookup.get(model, {}).get(count_key, 0) or 0)
            for model in models
        ]
        bars = ax.bar(
            x,
            counts,
            width=0.42,
            bottom=bottoms,
            color=color,
            edgecolor=HOUSE_COLORS["black"],
            linewidth=0.8,
            label=label,
        )
        for index, (bar, count) in enumerate(zip(bars, counts, strict=False)):
            if count <= 0.0:
                continue
            is_small_top_segment = category_index == len(categories) - 1 and count < 10.0
            text_color = "white" if label == "Correct" else HOUSE_COLORS["black"]
            label_y = bottoms[index] + count / 2
            label_text = f"{int(count)}\n{label.lower()}"
            va = "center"
            if is_small_top_segment:
                label_y = bottoms[index] + count + 2.2
                label_text = f"{int(count)} {label.lower()}"
                va = "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                label_text,
                ha="center",
                va=va,
                fontsize=8,
                color=text_color,
                fontweight="bold" if label == "Correct" else "normal",
                linespacing=1.05,
            )
        bottoms += np.array(counts, dtype=float)

    ax.set_title("Individual agent flag crop test (n=100)")
    ax.set_ylabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in models])
    max_n = max([int(lookup.get(model, {}).get("n", 0) or 0) for model in models] + [1])
    ax.set_ylim(0, max_n + 14)
    ax.set_xlim(-0.36, x[-1] + 0.36)
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    style_axis(ax, grid=False)
    ax.spines["left"].set_visible(False)


def draw_paired_compatibility_panel(
    ax: plt.Axes,
    pair_summary: dict[str, Any],
    *,
    models: list[str],
) -> None:
    rows = paired_compatibility_outcome_rows(pair_summary, models=models)
    labels = [
        str(row["label"]).replace("Both compatible", "Both\ncompatible").replace(
            "Both incompatible",
            "Both\nincompatible",
        )
        for row in rows
    ]
    counts = [int(row["count"]) for row in rows]
    colors = [
        HOUSE_COLORS["teal"],
        model_color(models[0], HOUSE_COLORS["blue"]),
        model_color(models[1], HOUSE_COLORS["red"]),
        HOUSE_COLORS["light_gray"],
    ]
    x = np.arange(len(rows), dtype=float)
    bars = ax.bar(
        x,
        counts,
        width=0.62,
        color=colors,
        edgecolor=HOUSE_COLORS["black"],
        linewidth=0.8,
    )
    for bar, count in zip(bars, counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(0.15, 0.025 * max(counts or [1])),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    p_value = float(pair_summary["mcnemar_exact_p"])
    ax.text(
        0.5,
        0.96,
        f"McNemar exact p={p_value:.3g}; pairs={int(pair_summary['n_pairs'])}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=HOUSE_COLORS["gray"],
    )
    ax.set_title("Matched-pair crop compatibility")
    ax.set_ylabel("Matched crops")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(counts + [1]) * 1.28 + 0.5)
    style_axis(ax, grid=True)


def draw_main_figure(
    *,
    models: list[str],
    model_visual_read_categories: list[dict[str, Any]],
    figsize: tuple[float, float] = (3.6, 4.6),
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_visual_read_category_panel(ax, model_visual_read_categories, models=models)
    fig.tight_layout()
    return fig


def wrapped(value: Any, *, width: int = 46) -> str:
    text = str(value or "").strip()
    if not text:
        return "(no reason recorded)"
    return textwrap.fill(text, width=width)


def draw_example_figure(
    examples: list[dict[str, Any]],
    *,
    models: list[str],
    title: str | None = "Same crop, different visual read",
) -> plt.Figure:
    row_count = max(1, len(examples))
    fig = plt.figure(figsize=(7.4, 2.35 * row_count))
    grid = fig.add_gridspec(
        row_count,
        3,
        width_ratios=[1.05, 1.55, 1.55],
        wspace=0.28,
        hspace=0.55,
    )

    if not examples:
        ax = fig.add_subplot(grid[:, :])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No qualitative example available: need a paired crop where GPT-4o gives a "
            "crop-compatible answer, GPT-5.4 gives a crop-incompatible answer, and "
            "crop_path exists.",
            ha="center",
            va="center",
            fontsize=9,
            color=HOUSE_COLORS["gray"],
            wrap=True,
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        return fig

    for row_index, example in enumerate(examples):
        image_ax = fig.add_subplot(grid[row_index, 0])
        left_ax = fig.add_subplot(grid[row_index, 1])
        right_ax = fig.add_subplot(grid[row_index, 2])

        image = plt.imread(example["crop_path"])
        image_ax.imshow(image)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color(HOUSE_COLORS["black"])
        label = str(example["crop_informativeness_label"])
        image_ax.set_title(
            f"Truth: {example['truth_country']}\n"
            f"{INFO_LABELS.get(label, label.title())} crop",
            fontsize=8.5,
            pad=5,
        )

        panels = [
            (
                left_ax,
                models[0],
                example["left_choice_country"],
                example["left_reason"],
                model_color(models[0], HOUSE_COLORS["blue"]),
            ),
            (
                right_ax,
                models[1],
                example["right_choice_country"],
                example["right_reason"],
                model_color(models[1], HOUSE_COLORS["red"]),
            ),
        ]
        for ax, model, choice, reason, color in panels:
            ax.axis("off")
            ax.text(
                0.0,
                1.0,
                f"{display_model(model)}\nAnswer: {choice or '<invalid>'}\n\n{wrapped(reason)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color=HOUSE_COLORS["black"],
                linespacing=1.23,
            )
            ax.plot(
                [0, 1],
                [1.04, 1.04],
                transform=ax.transAxes,
                color=color,
                lw=3,
                clip_on=False,
            )

    top = 0.9
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    else:
        top = 0.96
    fig.subplots_adjust(left=0.04, right=0.98, top=top, bottom=0.05)
    return fig


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / f"{stem}.png",
        figure_dir / f"{stem}.pdf",
        figure_dir / f"{stem}.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    data_dir: Path,
    stem: str,
    source: Path,
    models: list[str],
    pair_columns: list[str],
    pair_stats: dict[str, int],
    model_visual_read_categories: list[dict[str, Any]],
    manual_adjudication_rows: list[dict[str, Any]],
    manual_adjudication_source: Path | None,
    examples: list[dict[str, Any]],
    gallery_examples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        data_dir / f"{stem}_model_visual_read_categories.csv",
        model_visual_read_categories,
    )
    adjudication_path = data_dir / f"{stem}_manual_adjudication_template.csv"
    should_write_adjudication = True
    if adjudication_path.exists():
        should_write_adjudication = False
    if manual_adjudication_source is not None:
        try:
            if manual_adjudication_source.resolve() == adjudication_path.resolve():
                should_write_adjudication = False
        except FileNotFoundError:
            pass
    if should_write_adjudication:
        write_csv(adjudication_path, manual_adjudication_rows)
    write_csv(data_dir / f"{stem}_example_pairs.csv", examples)
    for stale_name in [
        f"{stem}_model_correctness.csv",
        f"{stem}_paired_discordance.csv",
        f"{stem}_paired_outcomes.csv",
        f"{stem}_model_crop_compatibility.csv",
        f"{stem}_paired_crop_compatibility.csv",
        f"{stem}_paired_crop_compatibility_outcomes.csv",
    ]:
        stale_path = data_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    summary = {
        "source_csv": str(source),
        "models": models,
        "n_rows_total": len(rows),
        "n_rows_by_model": dict(Counter(row["model"] for row in rows)),
        "pair_columns": pair_columns,
        "pair_stats": pair_stats,
        "model_visual_read_categories": model_visual_read_categories,
        "manual_adjudication_template": str(adjudication_path),
        "manual_adjudication_source": (
            str(manual_adjudication_source) if manual_adjudication_source else None
        ),
        "example_pairs": examples,
        "failure_gallery_examples": gallery_examples,
    }
    json_path = data_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return json_path


def main() -> None:
    args = parse_args()
    setup_paper_house_style()
    apply_paper_sans_rc()

    raw_rows = read_csv_rows(args.source)
    rows = normalize_rows(raw_rows)
    models = [args.left_model, args.right_model]
    analysis_rows = [row for row in rows if row["model"] in set(models)]
    if not analysis_rows:
        raise ValueError(
            f"No rows found for requested models: {args.left_model}, {args.right_model}"
        )

    pair_columns = infer_pair_columns(analysis_rows, args.pair_columns)
    pairs, pair_stats = build_pairs(
        analysis_rows,
        pair_columns=pair_columns,
        left_model=args.left_model,
        right_model=args.right_model,
    )
    if not pairs:
        raise ValueError(
            "No matched model pairs found. Check --left-model, --right-model, and "
            "--pair-columns."
        )

    manual_labels = (
        read_manual_adjudication(args.manual_adjudication, models=models)
        if args.manual_adjudication is not None
        else None
    )
    model_visual_read_categories = summarize_model_visual_read_categories(
        analysis_rows,
        models=models,
        manual_labels=manual_labels,
    )
    manual_adjudication_rows = paired_manual_adjudication_rows(
        pairs,
        models=models,
        source=args.source,
    )
    if args.max_examples < 1:
        raise ValueError("--max-examples must be >= 1")
    selected_pair_ids = parse_selected_pair_ids(args.selected_example_pair_ids)
    example_candidates = select_example_pairs(pairs, source=args.source, max_examples=None)
    gallery_examples = [dict(row) for row in example_candidates[: args.max_examples]]
    if selected_pair_ids:
        examples = selected_examples_by_pair_ids(example_candidates, selected_pair_ids)
    else:
        examples = [dict(row) for row in gallery_examples[:3]]
        for row in examples:
            row["paper_default_example"] = True

    example_ids = {str(row.get("pair_id")) for row in examples}
    for row in gallery_examples:
        row["paper_default_example"] = str(row.get("pair_id")) in example_ids

    main_fig = draw_main_figure(
        models=models,
        model_visual_read_categories=model_visual_read_categories,
    )
    main_half_fig = draw_main_figure(
        models=models,
        model_visual_read_categories=model_visual_read_categories,
        figsize=(3.6, 2.3),
    )
    example_fig = draw_example_figure(
        examples,
        models=models,
        title=None,
    )
    gallery_fig = draw_example_figure(
        gallery_examples,
        models=models,
        title=f"Candidate crop-incompatibility gallery (top {len(gallery_examples)})",
    )

    main_paths = save_figure(main_fig, args.figure_dir, f"{args.stem}_main")
    main_half_paths = save_figure(
        main_half_fig,
        args.figure_dir,
        f"{args.stem}_main_half_height",
    )
    example_paths = save_figure(example_fig, args.figure_dir, f"{args.stem}_example_pairs")
    gallery_paths = save_figure(gallery_fig, args.figure_dir, f"{args.stem}_failure_gallery")
    summary_path = write_outputs(
        data_dir=args.data_dir,
        stem=args.stem,
        source=args.source,
        models=models,
        pair_columns=pair_columns,
        pair_stats=pair_stats,
        model_visual_read_categories=model_visual_read_categories,
        manual_adjudication_rows=manual_adjudication_rows,
        manual_adjudication_source=args.manual_adjudication,
        examples=examples,
        gallery_examples=gallery_examples,
        rows=analysis_rows,
    )

    print("Wrote visual-only audit figures:")
    for path in [*main_paths, *main_half_paths, *example_paths, *gallery_paths]:
        print(f"  {path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
