#!/usr/bin/env python3
"""Estimate empirical Flag Game crop fields and draw a Mechanism 1 figure.

The preferred input is a repeated isolated crop-probe table. Each row is one
crop-only model answer. The script also supports a smoke-test path that reads
completed Flag Game run directories and uses their t=0 isolated probes.

Output files:
  paper/exports/data/<stem>_crop_fields.csv
  paper/exports/data/<stem>_target_model_summary.csv
  paper/exports/data/<stem>_coverage.csv
  paper/exports/data/<stem>_qsg_predictions.csv
  paper/exports/data/<stem>_summary.json
  paper/exports/figures/<stem>.{png,pdf,svg}
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"
DEFAULT_STEM = "flag_game_mechanism1_empirical_crop_fields"
DEFAULT_SOCIAL_SUMMARY = DATA_DIR / "flag_game_pairwise_n_scaling_main_summary.csv"
DEFAULT_MECHANISM_RUN_ROOT = (
    ROOT
    / "results"
    / "flag_game"
    / "proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5"
    / "04_model_mix_alpha_neutral"
)
DEFAULT_N_VALUES = [4, 8, 16, 32, 64, 128]

BLUE = "#2D8CFF"
BLUE_DARK = "#1764B5"
ORANGE = "#F17C2E"
ORANGE_DARK = "#B65116"
PURPLE = "#7C6FB6"
GREEN = "#00A36F"
GRAY = "#747C85"
LIGHT_GRAY = "#D9DEE5"
INK = "#25272B"
WHITE = "#FFFFFF"

FIELD_CMAP = LinearSegmentedColormap.from_list(
    "truth_positive_field",
    [ORANGE_DARK, "#F4F5F7", BLUE_DARK],
)

COUNTRY_COLUMNS = (
    "country",
    "predicted_country",
    "choice_country",
    "prediction_country",
    "answer_country",
    "final_country",
)
TARGET_COLUMNS = ("truth_country", "target_country", "true_country", "image_country")
MODEL_COLUMNS = ("model", "model_name")
VALID_COLUMNS = ("valid", "is_valid", "parsed", "parse_valid")
COUNTRY_LIST_COLUMNS = (
    "allowed_countries_json",
    "countries_json",
    "country_list_json",
    "allowed_countries",
    "countries",
)
EXPLICIT_CROP_KEY_COLUMNS = (
    "crop_key",
    "crop_id",
    "pair_id",
    "stimulus_id",
    "crop_path",
)


def add_arial() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def setup_style() -> None:
    add_arial()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 8.7,
            "axes.labelsize": 9.1,
            "axes.titlesize": 9.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.3,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", colors=INK, direction="out", top=False, right=False)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=LIGHT_GRAY, alpha=0.65, linewidth=0.7)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", color=INK, pad=4)


def setup_log_n_axis(ax: plt.Axes, n_values: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Population size N", labelpad=3)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def country_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).casefold())


def first_value(row: dict[str, Any], names: tuple[str, ...] | list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and normalize_text(value) != "":
            return value
    return None


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or normalize_text(value) == "":
        return default
    if isinstance(value, bool):
        return value
    text = normalize_text(value).casefold()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_int(value: Any) -> int | None:
    if value is None or normalize_text(value) == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value is None or normalize_text(value) == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_country_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    text = normalize_text(value)
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [normalize_text(item) for item in parsed if normalize_text(item)]
    return [part.strip() for part in text.split(",") if part.strip()]


def canonical_country(value: Any, countries: list[str]) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    lookup = {country_key(country): country for country in countries}
    return lookup.get(country_key(text), text)


def merge_countries(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, str] = {}
    for row in rows:
        for country in row.get("countries", []) or []:
            key = country_key(country)
            if key:
                seen.setdefault(key, normalize_text(country))
        for name in ("target", "choice"):
            country = row.get(name)
            key = country_key(country)
            if key:
                seen.setdefault(key, normalize_text(country))
    return sorted(seen.values())


def field_class(h_value: float, theta: float) -> str:
    if h_value > theta:
        return "truth"
    if h_value < -theta:
        return "rival"
    return "ambiguous"


def field_color(label: str) -> str:
    return {"truth": BLUE, "rival": ORANGE, "ambiguous": GRAY}.get(label, GRAY)


def value_from_any(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and normalize_text(value) != "":
            return value
    return None


def crop_geometry_from_row(row: dict[str, Any]) -> dict[str, float | None]:
    left = parse_float(value_from_any(row, ("crop_left", "left", "crop_x", "x")))
    top = parse_float(value_from_any(row, ("crop_top", "top", "crop_y", "y")))
    width = parse_float(value_from_any(row, ("crop_width", "width", "crop_w", "w")))
    height = parse_float(value_from_any(row, ("crop_height", "height", "crop_h", "h")))
    canvas_width = parse_float(value_from_any(row, ("canvas_width", "flag_width", "image_width")))
    canvas_height = parse_float(value_from_any(row, ("canvas_height", "flag_height", "image_height")))
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }


def infer_crop_key(row: dict[str, Any]) -> str:
    explicit = first_value(row, EXPLICIT_CROP_KEY_COLUMNS)
    if explicit is not None:
        return normalize_text(explicit)
    geom = crop_geometry_from_row(row)
    if all(geom[name] is not None for name in ("left", "top", "width", "height")):
        pool = normalize_text(row.get("country_pool") or row.get("pool"))
        target = normalize_text(first_value(row, TARGET_COLUMNS) or row.get("target"))
        return (
            f"{pool}|{target}|"
            f"{geom['top']:.6g}|{geom['left']:.6g}|{geom['height']:.6g}|{geom['width']:.6g}"
        )
    agent = first_value(row, ("agent_id", "crop_index"))
    seed = first_value(row, ("seed", "trial_seed"))
    target = normalize_text(first_value(row, TARGET_COLUMNS) or row.get("target"))
    if agent is not None and seed is not None:
        return f"{target}|seed={seed}|agent={agent}"
    raise ValueError(
        "Could not infer a crop key. Provide crop_id/crop_key or crop geometry columns."
    )


def normalize_probe_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    countries = parse_country_list(first_value(row, COUNTRY_LIST_COLUMNS))
    target = normalize_text(first_value(row, TARGET_COLUMNS))
    model = normalize_text(first_value(row, MODEL_COLUMNS)) or "unknown_model"
    choice = normalize_text(first_value(row, COUNTRY_COLUMNS))
    valid = parse_bool(first_value(row, VALID_COLUMNS), default=bool(choice))
    if countries:
        target = canonical_country(target, countries)
        choice = canonical_country(choice, countries)
    geom = crop_geometry_from_row(row)
    normalized = {
        "source": source,
        "model": model,
        "target": target,
        "choice": choice,
        "valid": valid and bool(choice),
        "countries": countries,
        "crop_key": infer_crop_key(row),
        "crop_index": parse_int(first_value(row, ("crop_index", "agent_id"))),
        "agent_id": parse_int(first_value(row, ("agent_id",))),
        "N": parse_int(first_value(row, ("N", "n", "population_size"))),
        "seed": parse_int(first_value(row, ("seed", "trial_seed"))),
        "country_pool": normalize_text(row.get("country_pool") or row.get("pool")),
        "truth_flag_path": normalize_text(row.get("truth_flag_path")),
        "crop_path": normalize_text(row.get("crop_path")),
        **geom,
    }
    return normalized


def read_table_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
    return rows


def load_probe_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_table_rows(path):
            rows.append(normalize_probe_row(row, source=str(path)))
    return rows


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_run_root_initial_probes(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        manifest_paths = [root / "trial_manifest.json"] if (root / "trial_manifest.json").exists() else list(
            root.rglob("trial_manifest.json")
        )
        for manifest_path in sorted(manifest_paths):
            trial_dir = manifest_path.parent
            if trial_dir in seen:
                continue
            seen.add(trial_dir)
            probes_path = trial_dir / "probes.jsonl"
            if not probes_path.exists():
                continue
            manifest = read_json(manifest_path)
            summary = read_json(trial_dir / "summary.json") if (trial_dir / "summary.json").exists() else {}
            countries = list(manifest.get("countries") or summary.get("countries") or [])
            truth = normalize_text(manifest.get("truth_country") or summary.get("truth_country"))
            country_pool = normalize_text(manifest.get("country_pool") or summary.get("country_pool"))
            n_value = int(summary.get("N") or len(manifest.get("assignments", [])) or 0)
            seed_value = summary.get("seed")
            if seed_value is None:
                seed_value = seed_from_path(trial_dir)
            assignment_by_agent = {
                int(item["agent_id"]): item for item in manifest.get("assignments", []) if "agent_id" in item
            }
            canvas = manifest.get("canvas") if isinstance(manifest.get("canvas"), dict) else {}
            truth_flag_path = trial_dir / "artifacts" / "truth_flag.png"

            for raw in read_table_rows(probes_path):
                if parse_int(raw.get("t")) not in (0, None):
                    continue
                agent_id = parse_int(raw.get("agent_id"))
                if agent_id is None:
                    continue
                assignment = assignment_by_agent.get(agent_id, {})
                row = {
                    "source": str(probes_path),
                    "model": normalize_text(raw.get("model")) or "unknown_model",
                    "target": truth,
                    "choice": canonical_country(raw.get("country"), countries),
                    "valid": parse_bool(raw.get("valid"), default=bool(raw.get("country"))),
                    "countries": countries,
                    "crop_key": (
                        f"{country_pool}|{truth}|"
                        f"{assignment.get('top')}|{assignment.get('left')}|"
                        f"{assignment.get('height')}|{assignment.get('width')}"
                    ),
                    "crop_index": parse_int(assignment.get("crop_index")),
                    "agent_id": agent_id,
                    "N": n_value,
                    "seed": parse_int(seed_value),
                    "country_pool": country_pool,
                    "truth_flag_path": str(truth_flag_path) if truth_flag_path.exists() else "",
                    "crop_path": str(trial_dir / "artifacts" / f"agent_{agent_id:02d}_crop.png"),
                    "left": parse_float(assignment.get("left")),
                    "top": parse_float(assignment.get("top")),
                    "width": parse_float(assignment.get("width")),
                    "height": parse_float(assignment.get("height")),
                    "canvas_width": parse_float(canvas.get("width")),
                    "canvas_height": parse_float(canvas.get("height")),
                }
                rows.append(row)
    return rows


def fill_missing_country_lists(rows: list[dict[str, Any]]) -> None:
    countries_by_target: dict[str, list[str]] = {}
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        countries_by_target[target] = merge_countries(target_rows)
    for row in rows:
        if not row.get("countries"):
            row["countries"] = countries_by_target.get(row["target"], [])
        if row.get("countries"):
            row["target"] = canonical_country(row["target"], row["countries"])
            row["choice"] = canonical_country(row["choice"], row["countries"])


def estimate_crop_fields(
    probe_rows: list[dict[str, Any]],
    *,
    smoothing_lambda: float,
    epsilon: float,
    theta: float,
    rival_country: str | None = None,
) -> list[dict[str, Any]]:
    fill_missing_country_lists(probe_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        if not row.get("target"):
            continue
        grouped[(row["model"], row["target"], row["crop_key"])].append(row)

    preliminary: list[dict[str, Any]] = []
    for (model, target, crop_key), rows in grouped.items():
        countries = merge_countries(rows)
        if not countries or country_key(target) not in {country_key(country) for country in countries}:
            continue
        countries = [canonical_country(country, countries) for country in countries]
        valid_choices = [
            canonical_country(row.get("choice"), countries)
            for row in rows
            if row.get("valid") and country_key(row.get("choice"))
        ]
        valid_choices = [
            choice for choice in valid_choices if country_key(choice) in {country_key(country) for country in countries}
        ]
        b_value = len(valid_choices)
        if b_value <= 0:
            continue
        counts = Counter(valid_choices)
        k_value = len(countries)
        denom = b_value + smoothing_lambda * k_value
        probabilities = {
            country: (float(counts.get(country, 0)) + smoothing_lambda) / denom
            for country in countries
        }
        exemplar = rows[0]
        preliminary.append(
            {
                "model": model,
                "target": canonical_country(target, countries),
                "crop_key": crop_key,
                "B": b_value,
                "K": k_value,
                "countries": countries,
                "counts": dict(counts),
                "probabilities": probabilities,
                "N_values": sorted({int(row["N"]) for row in rows if row.get("N") is not None}),
                "seed_values": sorted({int(row["seed"]) for row in rows if row.get("seed") is not None}),
                "source_count": len(rows),
                "crop_index": exemplar.get("crop_index"),
                "country_pool": exemplar.get("country_pool"),
                "truth_flag_path": exemplar.get("truth_flag_path"),
                "crop_path": exemplar.get("crop_path"),
                "left": exemplar.get("left"),
                "top": exemplar.get("top"),
                "width": exemplar.get("width"),
                "height": exemplar.get("height"),
                "canvas_width": exemplar.get("canvas_width"),
                "canvas_height": exemplar.get("canvas_height"),
            }
        )

    rivals: dict[tuple[str, str], str] = {}
    grouped_fields: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in preliminary:
        grouped_fields[(row["model"], row["target"])].append(row)
    for key, rows in grouped_fields.items():
        _model, target = key
        if rival_country:
            countries = merge_countries(rows)
            rival = canonical_country(rival_country, countries)
            if country_key(rival) not in {country_key(country) for country in countries}:
                raise SystemExit(
                    f"Requested --rival-country {rival_country!r} is not in the allowed country list "
                    f"for target {target!r}."
                )
            if country_key(rival) == country_key(target):
                raise SystemExit("--rival-country must differ from the target country.")
            rivals[key] = rival
            continue
        support = Counter()
        for row in rows:
            for country, probability in row["probabilities"].items():
                if country_key(country) != country_key(target):
                    support[country] += float(probability)
        if not support:
            continue
        rivals[key] = sorted(support, key=lambda country: (-support[country], country))[0]

    fields: list[dict[str, Any]] = []
    for row in preliminary:
        key = (row["model"], row["target"])
        rival = rivals.get(key)
        if rival is None:
            continue
        target = row["target"]
        probabilities = row["probabilities"]
        p_truth = float(probabilities.get(target, 0.0))
        p_rival = float(probabilities.get(rival, 0.0))
        h_value = math.log((p_truth + epsilon) / (p_rival + epsilon))
        fields.append(
            {
                **row,
                "rival": rival,
                "p_truth": p_truth,
                "p_rival": p_rival,
                "h": h_value,
                "field_class": field_class(h_value, theta),
            }
        )
    return fields


def summarize_target_models(fields: list[dict[str, Any]], theta: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fields:
        groups[(row["model"], row["target"])].append(row)
    for (model, target), items in sorted(groups.items()):
        n = len(items)
        labels = Counter(str(item["field_class"]) for item in items)
        h_values = np.array([float(item["h"]) for item in items], dtype=float)
        b_values = np.array([int(item["B"]) for item in items], dtype=float)
        rows.append(
            {
                "model": model,
                "target": target,
                "rival": items[0]["rival"],
                "n_crops": n,
                "total_valid_reps": int(np.sum(b_values)),
                "theta": theta,
                "w_truth": labels["truth"] / n if n else 0.0,
                "w_rival": labels["rival"] / n if n else 0.0,
                "w_ambiguous": labels["ambiguous"] / n if n else 0.0,
                "h_mean": float(np.mean(h_values)) if n else math.nan,
                "h_sd": float(np.std(h_values)) if n else math.nan,
                "B_min": int(np.min(b_values)) if n else 0,
                "B_median": float(np.median(b_values)) if n else math.nan,
                "B_max": int(np.max(b_values)) if n else 0,
            }
        )
    return rows


def coverage_rows(
    summary_row: dict[str, Any],
    n_values: list[int],
) -> list[dict[str, Any]]:
    w_truth = float(summary_row["w_truth"])
    w_rival = float(summary_row["w_rival"])
    rows: list[dict[str, Any]] = []
    for n_value in n_values:
        p_truth = 1.0 - (1.0 - w_truth) ** n_value
        p_rival = 1.0 - (1.0 - w_rival) ** n_value
        p_both = (
            1.0
            - (1.0 - w_truth) ** n_value
            - (1.0 - w_rival) ** n_value
            + max(0.0, 1.0 - w_truth - w_rival) ** n_value
        )
        w_ambiguous = max(0.0, 1.0 - w_truth - w_rival)
        p_none = w_ambiguous**n_value
        p_truth_only = (1.0 - w_rival) ** n_value - p_none
        p_rival_only = (1.0 - w_truth) ** n_value - p_none
        rows.append(
            {
                "model": summary_row["model"],
                "target": summary_row["target"],
                "rival": summary_row["rival"],
                "N": n_value,
                "w_truth": w_truth,
                "w_rival": w_rival,
                "w_ambiguous": w_ambiguous,
                "P_truth": p_truth,
                "P_rival": p_rival,
                "P_both": p_both,
                "P_none": p_none,
                "P_truth_only": p_truth_only,
                "P_rival_only": p_rival_only,
            }
        )
    return rows


def qsg_fixed_point(
    h_values: np.ndarray,
    *,
    beta: float,
    coupling: float,
    eta: float,
    max_iter: int = 1000,
    tolerance: float = 1e-10,
) -> tuple[float, float, np.ndarray]:
    m_value = 0.0
    response = np.tanh(beta * h_values)
    for _ in range(max_iter):
        response = np.tanh(beta * (h_values + coupling * m_value))
        target_m = float(np.mean(response))
        next_m = (1.0 - eta) * m_value + eta * target_m
        if abs(next_m - m_value) < tolerance:
            m_value = next_m
            break
        m_value = next_m
    response = np.tanh(beta * (h_values + coupling * m_value))
    q_value = float(np.mean(response * response))
    return float(np.mean(response)), q_value, response


def classify_qsg_endpoint(m_value: float, q_value: float, *, m_c: float, q_c: float) -> str:
    if q_value < q_c:
        return "fragmentation"
    if m_value > m_c:
        return "correct_consensus"
    if m_value < -m_c:
        return "wrong_consensus"
    return "polarization"


def empirical_qsg_rows(
    fields: list[dict[str, Any]],
    *,
    model: str,
    target: str,
    n_values: list[int],
    trials: int,
    seed: int,
    beta: float,
    coupling: float,
    eta: float,
    m_c: float,
    q_c: float,
) -> list[dict[str, Any]]:
    selected = [
        row for row in fields if row["model"] == model and country_key(row["target"]) == country_key(target)
    ]
    if not selected:
        return []
    h_pool = np.array([float(row["h"]) for row in selected], dtype=float)
    rival = str(selected[0]["rival"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for n_value in n_values:
        outcomes = Counter()
        m_values: list[float] = []
        q_values: list[float] = []
        truth_vote_shares: list[float] = []
        for _ in range(trials):
            h_sample = rng.choice(h_pool, size=n_value, replace=True)
            m_value, q_value, response = qsg_fixed_point(
                h_sample,
                beta=beta,
                coupling=coupling,
                eta=eta,
            )
            endpoint = classify_qsg_endpoint(m_value, q_value, m_c=m_c, q_c=q_c)
            outcomes[endpoint] += 1
            m_values.append(m_value)
            q_values.append(q_value)
            truth_vote_shares.append(float(np.mean(response >= 0.0)))
        row = {
            "model": model,
            "target": selected[0]["target"],
            "rival": rival,
            "N": n_value,
            "trials": trials,
            "mean_m": float(np.mean(m_values)),
            "mean_q": float(np.mean(q_values)),
            "mean_truth_vote_share": float(np.mean(truth_vote_shares)),
            "mean_rival_vote_share": 1.0 - float(np.mean(truth_vote_shares)),
        }
        for endpoint in ("correct_consensus", "wrong_consensus", "polarization", "fragmentation"):
            row[endpoint] = outcomes[endpoint] / trials if trials else 0.0
        rows.append(row)
    return rows


def rounded_geometry_key(values: dict[str, Any]) -> tuple[int, int, int, int] | None:
    parsed = {
        name: parse_float(values.get(name))
        for name in ("top", "left", "height", "width")
    }
    if any(value is None for value in parsed.values()):
        return None
    return tuple(int(round(float(parsed[name]))) for name in ("top", "left", "height", "width"))


def mechanism_seed_dir(run_root: Path, *, n_value: int, condition: str, seed: str) -> Path:
    return run_root / f"N{n_value}" / condition / seed


def load_mechanism_crop_sets(
    fields: list[dict[str, Any]],
    *,
    run_root: Path,
    condition: str,
    seed: str,
    n_values: list[int],
    theta: float,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    field_by_geometry: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in fields:
        key = rounded_geometry_key(row)
        if key is not None:
            field_by_geometry[key] = row

    rows: list[dict[str, Any]] = []
    missing_messages: list[str] = []
    for n_value in n_values:
        seed_dir = mechanism_seed_dir(
            run_root,
            n_value=n_value,
            condition=condition,
            seed=seed,
        )
        manifest_path = seed_dir / "trial_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing mechanism manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        assignments = list(manifest.get("assignments") or [])
        h_values: list[float] = []
        p_truth_values: list[float] = []
        p_rival_values: list[float] = []
        matched_items: list[dict[str, Any]] = []
        missing: list[tuple[int, int, int, int]] = []
        for assignment in assignments:
            key = rounded_geometry_key(assignment)
            if key is None or key not in field_by_geometry:
                if key is not None:
                    missing.append(key)
                continue
            field = field_by_geometry[key]
            h_values.append(float(field["h"]))
            p_truth_values.append(float(field["p_truth"]))
            p_rival_values.append(float(field["p_rival"]))
            matched_items.append({"assignment": assignment, "field": field})

        if missing:
            preview = ", ".join(str(item) for item in missing[:6])
            missing_messages.append(
                f"N={n_value}: missing {len(missing)} of {len(assignments)} assigned crops; first missing {preview}"
            )
        if not h_values:
            raise SystemExit(f"No measured crop fields matched {manifest_path}")

        h_array = np.array(h_values, dtype=float)
        labels = Counter(field_class(float(value), theta) for value in h_values)
        row = {
            "N": n_value,
            "condition": condition,
            "seed": seed,
            "n_agents": len(assignments),
            "n_matched": len(h_values),
            "n_missing": len(missing),
            "target": fields[0]["target"] if fields else "",
            "rival": fields[0]["rival"] if fields else "",
            "h_values": h_values,
            "p_truth_values": p_truth_values,
            "p_rival_values": p_rival_values,
            "matched_items": matched_items,
            "mean_h": float(np.mean(h_array)),
            "sd_h": float(np.std(h_array)),
            "frac_truth_field": labels["truth"] / len(h_values),
            "frac_rival_field": labels["rival"] / len(h_values),
            "frac_ambiguous_field": labels["ambiguous"] / len(h_values),
        }
        rows.append(row)

    if missing_messages and not allow_missing:
        message = "\n".join(missing_messages)
        raise SystemExit(
            "The mechanism crop sets are not fully covered by the measured crop-probe table.\n"
            f"{message}\n"
            "Rerun the probe collector with --crop-manifest for these N/seed manifests, "
            "or pass --allow-missing-mechanism-crops for a diagnostic-only figure."
        )
    return rows


def simulate_empirical_pairwise_h_values(
    h_values: np.ndarray,
    *,
    trials: int,
    seed: int,
    alpha: float,
    j_msg: float,
    beta: float,
    kappa: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_value = int(len(h_values))
    truth_shares: list[float] = []
    rival_shares: list[float] = []
    correct_consensus: list[float] = []
    wrong_consensus: list[float] = []
    polarization: list[float] = []
    fragmentation: list[float] = []

    for _ in range(trials):
        ell = h_values.astype(float).copy()
        for _step in range(kappa * n_value):
            speaker = int(rng.integers(n_value))
            listener = int(rng.integers(n_value - 1))
            if listener >= speaker:
                listener += 1
            speaker_pref = float(np.tanh(beta * ell[speaker]))
            message = 1.0 if rng.random() < (1.0 + speaker_pref) / 2.0 else -1.0
            target = float(h_values[listener]) + j_msg * message
            ell[listener] = (1.0 - alpha) * ell[listener] + alpha * target

        votes = np.where(np.tanh(beta * ell) >= 0.0, 1, -1)
        truth = float(np.mean(votes == 1))
        rival = 1.0 - truth
        truth_shares.append(truth)
        rival_shares.append(rival)
        is_correct_consensus = truth >= 0.85
        is_wrong_consensus = rival >= 0.85
        is_polarized = truth >= 0.25 and rival >= 0.25 and not is_correct_consensus and not is_wrong_consensus
        correct_consensus.append(float(is_correct_consensus))
        wrong_consensus.append(float(is_wrong_consensus))
        polarization.append(float(is_polarized))
        fragmentation.append(float(not is_correct_consensus and not is_wrong_consensus and not is_polarized))

    return {
        "mean_truth_vote_share": float(np.mean(truth_shares)),
        "mean_rival_vote_share": float(np.mean(rival_shares)),
        "correct_consensus": float(np.mean(correct_consensus)),
        "wrong_consensus": float(np.mean(wrong_consensus)),
        "polarization": float(np.mean(polarization)),
        "fragmentation": float(np.mean(fragmentation)),
    }


def empirical_qsg_rows_from_crop_sets(
    crop_sets: list[dict[str, Any]],
    *,
    trials: int,
    seed: int,
    beta: float,
    coupling: float,
    eta: float,
    alpha: float,
    j_msg: float,
    kappa: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, crop_set in enumerate(crop_sets):
        h_values = np.array(crop_set["h_values"], dtype=float)
        m_value, q_value, _response = qsg_fixed_point(
            h_values,
            beta=beta,
            coupling=coupling,
            eta=eta,
        )
        simulated = simulate_empirical_pairwise_h_values(
            h_values,
            trials=trials,
            seed=seed + 1009 * index,
            alpha=alpha,
            j_msg=j_msg,
            beta=beta,
            kappa=kappa,
        )
        rows.append(
            {
                "model": crop_set.get("condition", ""),
                "target": crop_set.get("target", ""),
                "rival": crop_set.get("rival", ""),
                "N": int(crop_set["N"]),
                "trials": trials,
                "mean_m": m_value,
                "mean_q": q_value,
                **simulated,
            }
        )
    return rows


def read_social_summary(path: Path, *, condition: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if condition and normalize_text(row.get("condition")) != condition:
                continue
            rows.append(
                {
                    "N": parse_int(row.get("N")),
                    "ciq": parse_float(row.get("final_mean")),
                    "iiq": parse_float(row.get("initial_mean")),
                    "polarization": parse_float(row.get("polarization_rate")),
                    "correct_consensus": parse_float(row.get("correct_consensus_rate")),
                }
            )
    return [row for row in rows if row["N"] is not None]


def load_endpoint_regime_summary(
    run_root: Path | None,
    *,
    condition: str,
    n_values: list[int],
) -> list[dict[str, Any]]:
    if run_root is None:
        return []
    rows: list[dict[str, Any]] = []
    for n_value in n_values:
        summary_path = run_root / f"N{n_value}" / "model_mix_summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if normalize_text(row.get("condition_name")) != normalize_text(condition):
                    continue
                rows.append(
                    {
                        "model": row.get("condition_name") or condition,
                        "target": "all",
                        "rival": "all",
                        "N": n_value,
                        "trials": parse_int(row.get("n_trials")) or 0,
                        "correct_consensus": parse_float(row.get("correct_consensus_rate")) or 0.0,
                        "wrong_consensus": parse_float(row.get("wrong_consensus_rate")) or 0.0,
                        "polarization": parse_float(row.get("polarization_rate")) or 0.0,
                        "fragmentation": parse_float(row.get("fragmentation_rate")) or 0.0,
                    }
                )
                break
    return rows


def field_order_rows_from_crop_sets(
    crop_sets: list[dict[str, Any]],
    *,
    beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for crop_set in crop_sets:
        h_values = np.array([float(value) for value in crop_set.get("h_values", [])], dtype=float)
        if h_values.size == 0:
            continue
        responses = np.tanh(beta * h_values)
        rows.append(
            {
                "N": int(crop_set["N"]),
                "target": crop_set.get("target", ""),
                "rival": crop_set.get("rival", ""),
                "n_agents": int(crop_set.get("n_agents") or h_values.size),
                "m0": float(np.mean(responses)),
                "q0": float(np.mean(responses * responses)),
            }
        )
    return sorted(rows, key=lambda row: int(row["N"]))


def field_order_rows_from_pool(
    fields: list[dict[str, Any]],
    *,
    n_values: list[int],
    beta: float,
) -> list[dict[str, Any]]:
    if not fields:
        return []
    responses = np.tanh(beta * np.array([float(row["h"]) for row in fields], dtype=float))
    m0 = float(np.mean(responses))
    q0 = float(np.mean(responses * responses))
    return [
        {
            "N": n_value,
            "target": fields[0].get("target", ""),
            "rival": fields[0].get("rival", ""),
            "n_agents": n_value,
            "m0": m0,
            "q0": q0,
        }
        for n_value in n_values
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def write_outputs(
    *,
    stem: str,
    fields: list[dict[str, Any]],
    target_summary: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    crop_sets: list[dict[str, Any]],
    field_order_rows: list[dict[str, Any]],
    qsg_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    source_note: str,
    figure_paths: list[Path],
) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    field_rows = []
    for row in fields:
        field_rows.append(
            {
                **row,
                "countries_json": json_dumps_compact(row.get("countries", [])),
                "counts_json": json_dumps_compact(row.get("counts", {})),
                "probabilities_json": json_dumps_compact(row.get("probabilities", {})),
                "N_values_json": json_dumps_compact(row.get("N_values", [])),
                "seed_values_json": json_dumps_compact(row.get("seed_values", [])),
            }
        )
    field_path = DATA_DIR / f"{stem}_crop_fields.csv"
    write_csv(
        field_path,
        field_rows,
        [
            "model",
            "target",
            "rival",
            "crop_key",
            "B",
            "K",
            "p_truth",
            "p_rival",
            "h",
            "field_class",
            "countries_json",
            "counts_json",
            "probabilities_json",
            "N_values_json",
            "seed_values_json",
            "source_count",
            "country_pool",
            "crop_index",
            "left",
            "top",
            "width",
            "height",
            "canvas_width",
            "canvas_height",
            "truth_flag_path",
            "crop_path",
        ],
    )
    summary_path = DATA_DIR / f"{stem}_target_model_summary.csv"
    write_csv(
        summary_path,
        target_summary,
        [
            "model",
            "target",
            "rival",
            "n_crops",
            "total_valid_reps",
            "theta",
            "w_truth",
            "w_rival",
            "w_ambiguous",
            "h_mean",
            "h_sd",
            "B_min",
            "B_median",
            "B_max",
        ],
    )
    coverage_path = DATA_DIR / f"{stem}_coverage.csv"
    write_csv(
        coverage_path,
        coverage,
        [
            "model",
            "target",
            "rival",
            "N",
            "w_truth",
            "w_rival",
            "w_ambiguous",
            "P_truth",
            "P_rival",
            "P_both",
            "P_none",
            "P_truth_only",
            "P_rival_only",
        ],
    )
    crop_sets_path = DATA_DIR / f"{stem}_mechanism_crop_sets.csv"
    crop_set_rows = []
    for row in crop_sets:
        h_values = [float(value) for value in row.get("h_values", [])]
        crop_set_rows.append(
            {
                "N": row.get("N"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "target": row.get("target"),
                "rival": row.get("rival"),
                "n_agents": row.get("n_agents"),
                "n_matched": row.get("n_matched"),
                "n_missing": row.get("n_missing"),
                "mean_h": row.get("mean_h"),
                "sd_h": row.get("sd_h"),
                "frac_truth_field": row.get("frac_truth_field"),
                "frac_rival_field": row.get("frac_rival_field"),
                "frac_ambiguous_field": row.get("frac_ambiguous_field"),
                "h_values_json": json_dumps_compact(h_values),
            }
        )
    write_csv(
        crop_sets_path,
        crop_set_rows,
        [
            "N",
            "condition",
            "seed",
            "target",
            "rival",
            "n_agents",
            "n_matched",
            "n_missing",
            "mean_h",
            "sd_h",
            "frac_truth_field",
            "frac_rival_field",
            "frac_ambiguous_field",
            "h_values_json",
        ],
    )
    field_order_path = DATA_DIR / f"{stem}_field_order.csv"
    write_csv(
        field_order_path,
        field_order_rows,
        [
            "N",
            "target",
            "rival",
            "n_agents",
            "m0",
            "q0",
        ],
    )
    qsg_path = DATA_DIR / f"{stem}_qsg_predictions.csv"
    write_csv(
        qsg_path,
        qsg_rows,
        [
            "model",
            "target",
            "rival",
            "N",
            "trials",
            "mean_m",
            "mean_q",
            "mean_truth_vote_share",
            "mean_rival_vote_share",
            "correct_consensus",
            "wrong_consensus",
            "polarization",
            "fragmentation",
        ],
    )
    endpoint_path = DATA_DIR / f"{stem}_endpoint_regimes.csv"
    write_csv(
        endpoint_path,
        endpoint_rows,
        [
            "model",
            "target",
            "rival",
            "N",
            "trials",
            "correct_consensus",
            "wrong_consensus",
            "polarization",
            "fragmentation",
        ],
    )
    metadata_path = DATA_DIR / f"{stem}_summary.json"
    payload = {
        "source_note": source_note,
        "n_probe_rows": len(probe_rows),
        "n_crop_fields": len(fields),
        "selected": selected,
        "outputs": {
            "crop_fields": str(field_path),
            "target_model_summary": str(summary_path),
            "coverage": str(coverage_path),
            "mechanism_crop_sets": str(crop_sets_path),
            "field_order": str(field_order_path),
            "qsg_predictions": str(qsg_path),
            "endpoint_regimes": str(endpoint_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return {
        "crop_fields": field_path,
        "target_model_summary": summary_path,
        "coverage": coverage_path,
        "mechanism_crop_sets": crop_sets_path,
        "field_order": field_order_path,
        "qsg_predictions": qsg_path,
        "endpoint_regimes": endpoint_path,
        "summary": metadata_path,
    }


def selected_fields(fields: list[dict[str, Any]], *, model: str, target: str) -> list[dict[str, Any]]:
    return [
        row for row in fields if row["model"] == model and country_key(row["target"]) == country_key(target)
    ]


def choose_default_selection(
    target_summary: list[dict[str, Any]],
    *,
    requested_model: str | None,
    requested_target: str | None,
) -> dict[str, Any]:
    candidates = target_summary
    if requested_model:
        candidates = [row for row in candidates if row["model"] == requested_model]
    if requested_target:
        candidates = [row for row in candidates if country_key(row["target"]) == country_key(requested_target)]
    if not candidates:
        raise SystemExit("No crop-field summary rows match the requested model/target.")
    candidates = sorted(
        candidates,
        key=lambda row: (
            int(row["target"] != "France"),
            -int(row["n_crops"]),
            row["model"],
            row["target"],
        ),
    )
    return candidates[0]


def draw_field_map(ax: plt.Axes, fields: list[dict[str, Any]], theta: float) -> None:
    drawable = [
        row
        for row in fields
        if all(row.get(name) is not None for name in ("left", "top", "width", "height"))
    ]
    if not drawable:
        ax.text(0.5, 0.5, "No crop geometry\nfor map panel", ha="center", va="center", color=GRAY)
        ax.set_xticks([])
        ax.set_yticks([])
        panel_title(ax, "A. Measured crop field")
        return

    canvas_width = next((float(row["canvas_width"]) for row in drawable if row.get("canvas_width")), None)
    canvas_height = next((float(row["canvas_height"]) for row in drawable if row.get("canvas_height")), None)
    if canvas_width is None:
        canvas_width = max(float(row["left"]) + float(row["width"]) for row in drawable)
    if canvas_height is None:
        canvas_height = max(float(row["top"]) + float(row["height"]) for row in drawable)

    truth_path = next(
        (
            Path(row["truth_flag_path"])
            for row in drawable
            if row.get("truth_flag_path") and Path(str(row["truth_flag_path"])).exists()
        ),
        None,
    )
    if truth_path is not None:
        image = plt.imread(truth_path)
        ax.imshow(image, extent=[0.0, canvas_width, canvas_height, 0.0], alpha=0.32, zorder=0)

    xs = [float(row["left"]) + 0.5 * float(row["width"]) for row in drawable]
    ys = [float(row["top"]) + 0.5 * float(row["height"]) for row in drawable]
    h_values = np.array([float(row["h"]) for row in drawable], dtype=float)
    vmax = max(theta * 2.0, float(np.nanpercentile(np.abs(h_values), 95)) if len(h_values) else theta)
    sizes = [17.0 + 8.0 * min(1.0, abs(float(row["h"])) / max(theta, 1e-9)) for row in drawable]
    scatter = ax.scatter(
        xs,
        ys,
        c=h_values,
        s=sizes,
        cmap=FIELD_CMAP,
        vmin=-vmax,
        vmax=vmax,
        edgecolors=WHITE,
        linewidths=0.45,
        alpha=0.92,
        zorder=3,
    )

    ax.set_xlim(0.0, canvas_width)
    ax.set_ylim(canvas_height, 0.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(LIGHT_GRAY)
        spine.set_linewidth(0.8)
    target = fields[0]["target"]
    rival = fields[0]["rival"]
    panel_title(ax, r"A. Measured crop field $h_i$")
    colorbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.060, pad=0.035)
    colorbar.set_label(r"$h_i$", labelpad=1, fontsize=6.6)
    colorbar.ax.tick_params(labelsize=6.2, width=0.7, length=2.2)
    colorbar.ax.text(
        0.0,
        -1.85,
        rival,
        transform=colorbar.ax.transAxes,
        ha="left",
        va="top",
        color=ORANGE_DARK,
        fontsize=6.3,
    )
    colorbar.ax.text(
        1.0,
        -1.85,
        target,
        transform=colorbar.ax.transAxes,
        ha="right",
        va="top",
        color=BLUE_DARK,
        fontsize=6.3,
    )


def h_example_label(row: dict[str, Any], theta: float) -> str:
    h_value = float(row["h"])
    if h_value > theta:
        return f"{row['target']}-supporting"
    if h_value < -theta:
        return f"{row['rival']}-supporting"
    return "ambiguous"


def choose_h_examples(fields: list[dict[str, Any]], theta: float) -> list[dict[str, Any]]:
    drawable = [row for row in fields if row.get("crop_path") and Path(str(row["crop_path"])).exists()]
    if not drawable:
        drawable = fields
    truth_rows = [row for row in drawable if float(row["h"]) > theta]
    rival_rows = [row for row in drawable if float(row["h"]) < -theta]
    ambiguous_rows = [row for row in drawable if abs(float(row["h"])) <= theta]
    examples: list[dict[str, Any]] = []
    if truth_rows:
        examples.append(max(truth_rows, key=lambda row: float(row["h"])))
    if rival_rows:
        examples.append(min(rival_rows, key=lambda row: float(row["h"])))
    if ambiguous_rows:
        examples.append(min(ambiguous_rows, key=lambda row: abs(float(row["h"]))))
    if len(examples) < 3:
        for row in sorted(drawable, key=lambda item: abs(float(item["h"])), reverse=True):
            if row not in examples:
                examples.append(row)
            if len(examples) == 3:
                break
    return examples[:3]


def draw_h_explainer(ax: plt.Axes, fields: list[dict[str, Any]], theta: float) -> None:
    if not fields:
        ax.text(0.5, 0.5, "No fields", ha="center", va="center", color=GRAY)
        panel_title(ax, "A. What h means")
        return
    target = str(fields[0]["target"])
    rival = str(fields[0]["rival"])
    examples = choose_h_examples(fields, theta)
    ax.set_axis_off()
    panel_title(ax, "A. What h means")
    ax.text(
        0.0,
        0.92,
        rf"$h_i=\log\,p_i({target})/p_i({rival})$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=8.0,
    )
    y_positions = [0.68, 0.42, 0.16]
    for row, y_pos in zip(examples, y_positions):
        crop_path = Path(str(row.get("crop_path") or ""))
        color = field_color(str(row["field_class"]))
        if crop_path.exists():
            inset = ax.inset_axes([0.0, y_pos - 0.085, 0.28, 0.16])
            inset.imshow(plt.imread(crop_path))
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.2)
        else:
            ax.add_patch(
                Rectangle(
                    (0.0, y_pos - 0.07),
                    0.25,
                    0.14,
                    transform=ax.transAxes,
                    facecolor="#F4F5F7",
                    edgecolor=color,
                    linewidth=1.2,
                )
            )
        p_truth = float(row["p_truth"])
        p_rival = float(row["p_rival"])
        h_value = float(row["h"])
        ax.text(
            0.34,
            y_pos + 0.045,
            h_example_label(row, theta),
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=color,
            fontsize=7.2,
            fontweight="bold",
        )
        ax.text(
            0.34,
            y_pos - 0.015,
            rf"$p_F={p_truth:.2f}$  $p_P={p_rival:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=INK,
            fontsize=6.8,
        )
        ax.text(
            0.34,
            y_pos - 0.073,
            rf"$h={h_value:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=INK,
            fontsize=6.8,
        )


def draw_h_distribution(ax: plt.Axes, fields: list[dict[str, Any]], theta: float) -> None:
    h_values = np.array([float(row["h"]) for row in fields], dtype=float)
    if len(h_values) == 0:
        ax.text(0.5, 0.5, "No fields", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Measured h distribution")
        return
    bins = np.linspace(
        min(float(np.min(h_values)), -2.5 * theta),
        max(float(np.max(h_values)), 2.5 * theta),
        20,
    )
    ax.hist(h_values, bins=bins, color=LIGHT_GRAY, edgecolor=WHITE, linewidth=0.7)
    ax.axvline(theta, color=BLUE, linewidth=1.2, linestyle="--")
    ax.axvline(-theta, color=ORANGE, linewidth=1.2, linestyle="--")
    ax.axvline(0.0, color=INK, linewidth=0.8, alpha=0.6)
    labels = Counter(row["field_class"] for row in fields)
    total = len(fields)
    text = (
        f"truth {labels['truth'] / total:.2f}\n"
        f"rival {labels['rival'] / total:.2f}\n"
        f"ambig {labels['ambiguous'] / total:.2f}"
    )
    ax.text(
        0.98,
        0.94,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=7.6,
    )
    ax.set_xlabel("h = log p(T)/p(R)")
    ax.set_ylabel("Crops")
    panel_title(ax, "B. Measured h distribution")
    style_axis(ax)


def draw_coverage(
    ax: plt.Axes,
    coverage: list[dict[str, Any]],
    n_values: list[int],
) -> None:
    x = np.array([int(row["N"]) for row in coverage], dtype=int)
    p_truth = np.array([float(row["P_truth"]) for row in coverage], dtype=float)
    p_rival = np.array([float(row["P_rival"]) for row in coverage], dtype=float)
    p_both = np.array([float(row["P_both"]) for row in coverage], dtype=float)
    first = coverage[0] if coverage else {}
    target = str(first.get("target") or "T")
    rival = str(first.get("rival") or "R")
    ax.plot(
        x,
        p_truth,
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        linewidth=1.55,
        label=rf"$P_T$ any {target}",
    )
    ax.plot(
        x,
        p_rival,
        color=ORANGE,
        marker="s",
        markerfacecolor=WHITE,
        linewidth=1.55,
        label=rf"$P_R$ any {rival}",
    )
    ax.plot(
        x,
        p_both,
        color=PURPLE,
        marker="^",
        markerfacecolor=WHITE,
        linewidth=1.55,
        label=r"$P_{\mathrm{both}}$ both",
    )
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Probability")
    panel_title(ax, "B. Field coverage by $N$")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8, handlelength=1.6)


def draw_initial_field_order(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    n_values: list[int],
) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No field-order rows", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Initial field balance")
        return
    rows = sorted(rows, key=lambda row: int(row["N"]))
    x = np.array([int(row["N"]) for row in rows], dtype=int)
    m_values = np.array([float(row["m0"]) for row in rows], dtype=float)
    q_values = np.array([float(row["q0"]) for row in rows], dtype=float)
    ax.plot(
        x,
        m_values,
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        linewidth=1.65,
        label=r"$m_0$ initial lean",
    )
    ax.plot(
        x,
        q_values,
        color=PURPLE,
        marker="s",
        markerfacecolor=WHITE,
        linewidth=1.55,
        label=r"$q_0$ field strength",
    )
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.5)
    ax.axhspan(0.55, 1.0, color=PURPLE, alpha=0.06, zorder=0)
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(-1.02, 1.02)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylabel("Order parameter")
    panel_title(ax, "B. Initial field balance")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8, handlelength=1.6)


def draw_group_evidence_states(ax: plt.Axes, coverage: list[dict[str, Any]]) -> None:
    if not coverage:
        ax.text(0.5, 0.5, "No coverage rows", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Evidence states")
        return
    rows = sorted(coverage, key=lambda row: int(row["N"]))
    target = str(rows[0]["target"])
    rival = str(rows[0]["rival"])
    x_positions = np.arange(len(rows), dtype=float)
    state_specs = [
        ("P_none", "none", LIGHT_GRAY),
        ("P_truth_only", f"{target} only", BLUE),
        ("P_rival_only", f"{rival} only", ORANGE),
        ("P_both", "both signs", PURPLE),
    ]
    bottom = np.zeros(len(rows), dtype=float)
    for key, label, color in state_specs:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=color,
            edgecolor=WHITE,
            linewidth=0.55,
            label=label,
        )
        bottom += values
    first = rows[0]
    w_truth = float(first["w_truth"])
    w_rival = float(first["w_rival"])
    w_ambiguous = float(first["w_ambiguous"])
    ax.text(
        0.02,
        0.98,
        f"per-crop: + {w_truth:.2f}  - {w_rival:.2f}  0 {w_ambiguous:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=7.2,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Probability")
    panel_title(ax, "B. Evidence states")
    style_axis(ax)
    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=5.9,
        handlelength=0.9,
        labelspacing=0.20,
        borderaxespad=0.1,
    )


def h_bucket_specs(theta: float) -> list[tuple[float, float, str, Any, str]]:
    return [
        (-math.inf, -2.0 * theta, r"$h<-2\theta$", FIELD_CMAP(0.04), "o"),
        (-2.0 * theta, -theta, r"$-2\theta\leq h<-\theta$", FIELD_CMAP(0.23), "s"),
        (-theta, theta, r"$|h|\leq\theta$", FIELD_CMAP(0.50), "^"),
        (theta, 2.0 * theta, r"$\theta<h\leq2\theta$", FIELD_CMAP(0.77), "D"),
        (2.0 * theta, math.inf, r"$h>2\theta$", FIELD_CMAP(0.96), "P"),
    ]


def h_bucket_story_labels(target: str, rival: str) -> list[str]:
    return [
        f"{rival} strong",
        f"{rival} weak",
        "ambiguous",
        f"{target} weak",
        f"{target} strong",
    ]


def draw_h_bucket_coverage(
    ax: plt.Axes,
    fields: list[dict[str, Any]],
    n_values: list[int],
    theta: float,
) -> None:
    if not fields:
        ax.text(0.5, 0.5, "No fields", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. h-bucket coverage")
        return
    h_values = np.array([float(row["h"]) for row in fields], dtype=float)
    x = np.array(n_values, dtype=int)
    total = len(h_values)
    for lower, upper, label, color, marker in h_bucket_specs(theta):
        if math.isinf(lower):
            mask = h_values < upper
        elif math.isinf(upper):
            mask = h_values > lower
        else:
            mask = (h_values >= lower) & (h_values <= upper)
        mass = float(np.mean(mask)) if total else 0.0
        y = 1.0 - (1.0 - mass) ** x
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markerfacecolor=WHITE,
            markeredgecolor=color,
            markeredgewidth=1.05,
            linewidth=1.45,
            label=f"{label} ({mass:.2f})",
        )
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("P(group samples bucket)")
    panel_title(ax, "B. h-bucket coverage")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=5.6, handlelength=1.4, labelspacing=0.12)


def h_bucket_index(value: float, theta: float) -> int:
    if value < -2.0 * theta:
        return 0
    if value < -theta:
        return 1
    if value <= theta:
        return 2
    if value <= 2.0 * theta:
        return 3
    return 4


def draw_extreme_h_bucket_distribution(
    ax: plt.Axes,
    fields: list[dict[str, Any]],
    n_values: list[int],
    theta: float,
    *,
    trials: int = 20000,
    seed: int = 20260502,
) -> None:
    if not fields:
        ax.text(0.5, 0.5, "No fields", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Most extreme sampled h")
        return
    h_pool = np.array([float(row["h"]) for row in fields], dtype=float)
    target = str(fields[0].get("target") or "truth")
    rival = str(fields[0].get("rival") or "rival")
    rng = np.random.default_rng(seed)
    bucket_specs = h_bucket_specs(theta)
    bucket_labels = h_bucket_story_labels(target, rival)
    values_by_bucket = np.zeros((len(bucket_specs), len(n_values)), dtype=float)
    for n_index, n_value in enumerate(n_values):
        sample = rng.choice(h_pool, size=(trials, n_value), replace=True)
        strongest = sample[np.arange(trials), np.argmax(np.abs(sample), axis=1)]
        bucket_ids = np.array([h_bucket_index(float(value), theta) for value in strongest], dtype=int)
        counts = np.bincount(bucket_ids, minlength=len(bucket_specs))
        values_by_bucket[:, n_index] = counts / float(trials)

    x_positions = np.arange(len(n_values), dtype=float)
    bottom = np.zeros(len(n_values), dtype=float)
    for bucket_index, (_lower, _upper, _label, color, _marker) in enumerate(bucket_specs):
        ax.bar(
            x_positions,
            values_by_bucket[bucket_index],
            bottom=bottom,
            color=color,
            edgecolor=WHITE,
            linewidth=0.55,
            label=bucket_labels[bucket_index],
        )
        bottom += values_by_bucket[bucket_index]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n_value) for n_value in n_values])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Fraction of groups")
    panel_title(ax, "B. Strongest crop by N")
    style_axis(ax)
    ax.text(
        0.03,
        0.96,
        "strong: >=4:1\nweak: 2-4:1",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=6.2,
        linespacing=1.05,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=5.6, handlelength=0.9, labelspacing=0.12)


def draw_mechanism_h_buckets(
    ax: plt.Axes,
    crop_sets: list[dict[str, Any]],
    theta: float,
) -> None:
    if not crop_sets:
        ax.text(0.5, 0.5, "No N-specific crop sets", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Initial h fields by N")
        return
    rows = sorted(crop_sets, key=lambda row: int(row["N"]))
    bucket_specs = h_bucket_specs(theta)
    target = str(rows[0].get("target") or "truth")
    rival = str(rows[0].get("rival") or "rival")
    labels = h_bucket_story_labels(target, rival)
    values_by_bucket = np.zeros((len(bucket_specs), len(rows)), dtype=float)
    for row_index, row in enumerate(rows):
        h_values = np.array(row["h_values"], dtype=float)
        bucket_ids = np.array([h_bucket_index(float(value), theta) for value in h_values], dtype=int)
        counts = np.bincount(bucket_ids, minlength=len(bucket_specs))
        values_by_bucket[:, row_index] = counts / max(1, len(h_values))

    x_positions = np.arange(len(rows), dtype=float)
    bottom = np.zeros(len(rows), dtype=float)
    for bucket_index, (_lower, _upper, _math_label, color, _marker) in enumerate(bucket_specs):
        ax.bar(
            x_positions,
            values_by_bucket[bucket_index],
            bottom=bottom,
            color=color,
            edgecolor=WHITE,
            linewidth=0.55,
            label=labels[bucket_index],
        )
        bottom += values_by_bucket[bucket_index]
    missing = [int(row.get("n_missing", 0)) for row in rows]
    if any(missing):
        for x_pos, row, count in zip(x_positions, rows, missing, strict=True):
            ax.text(
                x_pos,
                1.015,
                f"-{count}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=GRAY,
            )
    ax.text(
        0.02,
        0.98,
        "buckets use 2:1 odds",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=6.7,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.08 if any(missing) else 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Share of agents")
    panel_title(ax, "B. Initial h fields by N")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=5.35, handlelength=0.8, labelspacing=0.10)


def draw_mean_field(ax: plt.Axes, rows: list[dict[str, Any]], n_values: list[int]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No QSG predictions", ha="center", va="center", color=GRAY)
        panel_title(ax, "C. Mean-field resolution")
        return
    x = np.array([int(row["N"]) for row in rows], dtype=int)
    m_values = np.array([float(row["mean_m"]) for row in rows], dtype=float)
    q_values = np.array([float(row["mean_q"]) for row in rows], dtype=float)
    ax.plot(x, m_values, color=BLUE_DARK, marker="o", markerfacecolor=WHITE, linewidth=1.55, label=r"$m^*$ lean")
    ax.plot(x, q_values, color=PURPLE, marker="s", markerfacecolor=WHITE, linewidth=1.55, label=r"$q^*$ confidence")
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55, zorder=1)
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(-0.1, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Order parameter")
    panel_title(ax, "C. Mean-field resolution")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handlelength=1.7)


def draw_qsg_vote_shares(ax: plt.Axes, rows: list[dict[str, Any]], n_values: list[int]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No QSG predictions", ha="center", va="center", color=GRAY)
        panel_title(ax, "D. Finite-agent votes")
        return
    x = np.array([int(row["N"]) for row in rows], dtype=int)
    truth_share = np.array([float(row["mean_truth_vote_share"]) for row in rows], dtype=float)
    rival_share = np.array([float(row["mean_rival_vote_share"]) for row in rows], dtype=float)
    ax.plot(x, truth_share, color=BLUE, marker="o", markerfacecolor=WHITE, linewidth=1.55, label="truth vote share")
    ax.plot(x, rival_share, color=ORANGE, marker="s", markerfacecolor=WHITE, linewidth=1.55, label="rival vote share")
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Mean share")
    panel_title(ax, "D. Finite-agent votes")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handlelength=1.7)


def draw_simulated_vote_lines(ax: plt.Axes, rows: list[dict[str, Any]], n_values: list[int]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No vote predictions", ha="center", va="center", color=GRAY)
        panel_title(ax, "C. Simulated vote shares")
        return
    x = np.array([int(row["N"]) for row in rows], dtype=int)
    truth_share = np.array([float(row["mean_truth_vote_share"]) for row in rows], dtype=float)
    rival_share = np.array([float(row["mean_rival_vote_share"]) for row in rows], dtype=float)
    target = str(rows[0].get("target") or "Truth")
    rival = str(rows[0].get("rival") or "Rival")
    ax.plot(
        x,
        truth_share,
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        linewidth=1.65,
        label=f"{target} vote",
    )
    ax.plot(
        x,
        rival_share,
        color=ORANGE,
        marker="s",
        markerfacecolor=WHITE,
        linewidth=1.55,
        label=f"{rival} vote",
    )
    ax.axhline(0.5, color=INK, linewidth=0.8, alpha=0.40)
    for threshold in (0.25, 0.85):
        ax.axhline(threshold, color=GRAY, linewidth=0.85, linestyle="--", alpha=0.72)
    ax.text(
        1.01,
        0.85,
        "consensus",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color=GRAY,
        clip_on=False,
    )
    ax.text(
        1.01,
        0.25,
        "split",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color=GRAY,
        clip_on=False,
    )
    setup_log_n_axis(ax, n_values)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Mean final share")
    panel_title(ax, "C. Simulated vote shares")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=6.7, handlelength=1.6)


def draw_phase_plane(ax: plt.Axes, coverage: list[dict[str, Any]]) -> None:
    x = np.array([float(row["P_truth"]) for row in coverage], dtype=float)
    y = np.array([float(row["P_both"]) for row in coverage], dtype=float)
    n_values = [int(row["N"]) for row in coverage]
    ax.axvspan(0.0, 0.55, color=LIGHT_GRAY, alpha=0.4, zorder=0)
    ax.axhspan(0.55, 1.0, color=PURPLE, alpha=0.10, zorder=0)
    ax.axvspan(0.55, 1.0, ymin=0.0, ymax=0.55, color=GREEN, alpha=0.08, zorder=0)
    ax.plot(x, y, color=INK, linewidth=1.25, zorder=2)
    ax.scatter(x, y, s=28, color=BLUE_DARK, edgecolors=WHITE, linewidth=0.6, zorder=3)
    label_groups: dict[tuple[float, float], list[tuple[int, float, float]]] = defaultdict(list)
    for idx, n_value in enumerate(n_values):
        label_groups[(round(float(x[idx]), 2), round(float(y[idx]), 2))].append(
            (n_value, float(x[idx]), float(y[idx]))
        )
    crowded_corner_labels = 0
    for grouped in sorted(label_groups.values(), key=lambda values: min(item[0] for item in values)):
        grouped_ns = [item[0] for item in grouped]
        label = (
            f"{min(grouped_ns)}-{max(grouped_ns)}"
            if len(grouped_ns) > 2
            else ",".join(str(value) for value in grouped_ns)
        )
        label_x = float(np.mean([item[1] for item in grouped]))
        label_y = float(np.mean([item[2] for item in grouped]))
        if label_x > 0.94 and label_y > 0.94:
            label_x = 0.91
            label_y = 0.98 - 0.045 * crowded_corner_labels
            crowded_corner_labels += 1
        else:
            label_x = min(0.985, label_x + 0.015)
            label_y = min(0.985, label_y + 0.015)
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=7.0,
            color=INK,
        )
    if len(x) >= 2:
        ax.annotate(
            "",
            xy=(x[-1], y[-1]),
            xytext=(x[0], y[0]),
            arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": INK},
        )
    ax.text(0.08, 0.13, "coverage\nfailure", color=GRAY, fontsize=7.5)
    ax.text(0.62, 0.18, "correct\nconsensus", color=GREEN, fontsize=7.5)
    ax.text(0.60, 0.72, "polarization\nrisk", color=PURPLE, fontsize=7.5)
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("P_truth(N)")
    ax.set_ylabel("P_both(N)")
    panel_title(ax, "D. Population phase path")
    style_axis(ax, grid_axis="both")


def draw_qsg(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No QSG predictions", ha="center", va="center", color=GRAY)
        panel_title(ax, "E. Endpoint classes")
        return
    endpoint_specs = [
        ("correct_consensus", "Correct", GREEN),
        ("wrong_consensus", "Wrong", ORANGE),
        ("polarization", "Polarization", PURPLE),
        ("fragmentation", "Fragmentation", LIGHT_GRAY),
    ]
    x_positions = np.arange(len(rows), dtype=float)
    bottom = np.zeros(len(rows), dtype=float)
    for key, _label, color in endpoint_specs:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(x_positions, values, bottom=bottom, color=color, edgecolor=WHITE, linewidth=0.55)
        bottom += values
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Fraction")
    panel_title(ax, "E. Endpoint classes")
    style_axis(ax)
    handles = [Patch(facecolor=color, edgecolor=WHITE, label=label) for _key, label, color in endpoint_specs]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=5.9,
        handlelength=0.9,
        labelspacing=0.20,
        borderaxespad=0.1,
    )


def draw_endpoint_regimes(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No endpoint predictions", ha="center", va="center", color=GRAY)
        panel_title(ax, "D. Endpoint regimes")
        return
    endpoint_specs = [
        ("correct_consensus", "Correct", GREEN),
        ("wrong_consensus", "Wrong", ORANGE),
        ("polarization", "Polarization", PURPLE),
        ("fragmentation", "Fragmentation", LIGHT_GRAY),
    ]
    x_positions = np.arange(len(rows), dtype=float)
    bottom = np.zeros(len(rows), dtype=float)
    for key, _label, color in endpoint_specs:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(x_positions, values, bottom=bottom, color=color, edgecolor=WHITE, linewidth=0.55)
        bottom += values
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Fraction of runs")
    panel_title(ax, "D. Endpoint regimes across seeds")
    style_axis(ax)
    handles = [Patch(facecolor=color, edgecolor=WHITE, label=label) for _key, label, color in endpoint_specs]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.28),
        ncol=2,
        fontsize=5.8,
        handlelength=0.9,
        columnspacing=0.7,
        labelspacing=0.15,
        borderaxespad=0.0,
    )


def draw_seed_field_mix(ax: plt.Axes, crop_sets: list[dict[str, Any]]) -> None:
    if not crop_sets:
        ax.text(0.5, 0.5, "No seed crop fields", ha="center", va="center", color=GRAY)
        panel_title(ax, "E. Seed field mix")
        return
    rows = sorted(crop_sets, key=lambda row: int(row["N"]))
    target = str(rows[0].get("target") or "truth")
    rival = str(rows[0].get("rival") or "rival")
    specs = [
        ("frac_rival_field", f"{rival}-supporting", ORANGE),
        ("frac_ambiguous_field", "Ambiguous", LIGHT_GRAY),
        ("frac_truth_field", f"{target}-supporting", BLUE),
    ]
    x_positions = np.arange(len(rows), dtype=float)
    bottom = np.zeros(len(rows), dtype=float)
    for key, _label, color in specs:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=color,
            edgecolor=WHITE,
            linewidth=0.55,
        )
        bottom += values
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Share of agents")
    panel_title(ax, "E. Seed h composition")
    style_axis(ax)
    handles = [Patch(facecolor=color, edgecolor=WHITE, label=label) for _key, label, color in specs]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=5.75,
        handlelength=0.9,
        labelspacing=0.16,
        borderaxespad=0.1,
    )


def draw_seed_fields_with_votes(
    ax: plt.Axes,
    crop_sets: list[dict[str, Any]],
    qsg_rows: list[dict[str, Any]],
) -> None:
    if not crop_sets or not qsg_rows:
        ax.text(0.5, 0.5, "No seed mechanism rows", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Measured fields and votes")
        return
    crop_rows = sorted(crop_sets, key=lambda row: int(row["N"]))
    qsg_by_n = {int(row["N"]): row for row in qsg_rows}
    rows = [row for row in crop_rows if int(row["N"]) in qsg_by_n]
    if not rows:
        ax.text(0.5, 0.5, "No matched N values", ha="center", va="center", color=GRAY)
        panel_title(ax, "B. Measured fields and votes")
        return

    target = str(rows[0].get("target") or "truth")
    rival = str(rows[0].get("rival") or "rival")
    x_positions = np.arange(len(rows), dtype=float)
    bar_specs = [
        ("frac_rival_field", f"{rival} field", ORANGE),
        ("frac_ambiguous_field", "Ambiguous", LIGHT_GRAY),
        ("frac_truth_field", f"{target} field", BLUE),
    ]
    bottom = np.zeros(len(rows), dtype=float)
    for key, _label, color in bar_specs:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=color,
            alpha=0.42 if color != LIGHT_GRAY else 0.70,
            edgecolor=WHITE,
            linewidth=0.55,
            width=0.72,
        )
        bottom += values

    truth_vote = np.array(
        [float(qsg_by_n[int(row["N"])]["mean_truth_vote_share"]) for row in rows],
        dtype=float,
    )
    rival_vote = np.array(
        [float(qsg_by_n[int(row["N"])]["mean_rival_vote_share"]) for row in rows],
        dtype=float,
    )
    ax.plot(
        x_positions,
        truth_vote,
        color=BLUE_DARK,
        marker="o",
        markerfacecolor=WHITE,
        markeredgewidth=1.15,
        linewidth=1.65,
        label=f"{target} simulated vote",
        zorder=5,
    )
    ax.text(
        x_positions[-1] + 0.12,
        truth_vote[-1],
        f"{target} vote",
        ha="left",
        va="center",
        color=BLUE_DARK,
        fontsize=6.4,
    )
    ax.plot(
        x_positions,
        rival_vote,
        color=ORANGE_DARK,
        marker="s",
        markerfacecolor=WHITE,
        markeredgewidth=1.15,
        linewidth=1.55,
        label=f"{rival} simulated vote",
        zorder=5,
    )
    ax.text(
        x_positions[-1] + 0.12,
        rival_vote[-1],
        f"{rival} vote",
        ha="left",
        va="center",
        color=ORANGE_DARK,
        fontsize=6.4,
    )
    ax.axhline(0.5, color=INK, linewidth=0.75, alpha=0.45, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(row["N"])) for row in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Share")
    panel_title(ax, "B. Measured fields drive simulated votes")
    style_axis(ax)

    ax.set_xlim(-0.65, x_positions[-1] + 0.92)
    handles = [
        Patch(facecolor=ORANGE, alpha=0.42, edgecolor=WHITE, label=f"{rival}-supporting h"),
        Patch(facecolor=LIGHT_GRAY, alpha=0.70, edgecolor=WHITE, label="ambiguous h"),
        Patch(facecolor=BLUE, alpha=0.42, edgecolor=WHITE, label=f"{target}-supporting h"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.49, -0.36),
        ncol=3,
        fontsize=5.95,
        handlelength=1.0,
        columnspacing=0.9,
        labelspacing=0.10,
        borderaxespad=0.0,
    )


def draw_figure(
    *,
    stem: str,
    fields: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    crop_sets: list[dict[str, Any]],
    field_order_rows: list[dict[str, Any]],
    qsg_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    n_values: list[int],
    theta: float,
    mechanism_seed: str | None,
    figure_layout: str,
) -> list[Path]:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if figure_layout == "two_panel":
        fig = plt.figure(figsize=(6.65, 2.82), constrained_layout=False)
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.12, 1.0],
            left=0.068,
            right=0.992,
            top=0.82,
            bottom=0.300,
            wspace=0.36,
        )
        ax_map = fig.add_subplot(grid[0, 0])
        ax_mechanism = fig.add_subplot(grid[0, 1])
        draw_field_map(ax_map, fields, theta)
        draw_seed_fields_with_votes(ax_mechanism, crop_sets, qsg_rows)
        figure_paths = [
            FIGURE_DIR / f"{stem}.png",
            FIGURE_DIR / f"{stem}.pdf",
            FIGURE_DIR / f"{stem}.svg",
        ]
        for path in figure_paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return figure_paths

    if figure_layout == "mechanism_story":
        fig = plt.figure(figsize=(10.0, 3.05), constrained_layout=False)
        grid = fig.add_gridspec(
            1,
            4,
            width_ratios=[1.18, 1.02, 1.0, 1.0],
            left=0.055,
            right=0.995,
            top=0.82,
            bottom=0.335,
            wspace=0.36,
        )
        axes = {
            "map": fig.add_subplot(grid[0, 0]),
            "coverage": fig.add_subplot(grid[0, 1]),
            "votes": fig.add_subplot(grid[0, 2]),
            "endpoints": fig.add_subplot(grid[0, 3]),
        }
        draw_field_map(axes["map"], fields, theta)
        draw_initial_field_order(axes["coverage"], field_order_rows, n_values)
        draw_simulated_vote_lines(axes["votes"], qsg_rows, n_values)
        draw_endpoint_regimes(axes["endpoints"], endpoint_rows or qsg_rows)
        figure_paths = [
            FIGURE_DIR / f"{stem}.png",
            FIGURE_DIR / f"{stem}.pdf",
            FIGURE_DIR / f"{stem}.svg",
        ]
        for path in figure_paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return figure_paths

    fig = plt.figure(figsize=(12.2, 2.62), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.18, 1.02, 1.0, 1.0, 1.0],
        left=0.045,
        right=0.995,
        top=0.84,
        bottom=0.245,
        wspace=0.34,
    )
    axes = {
        "map": fig.add_subplot(grid[0, 0]),
        "states": fig.add_subplot(grid[0, 1]),
        "mean_field": fig.add_subplot(grid[0, 2]),
        "votes": fig.add_subplot(grid[0, 3]),
        "qsg": fig.add_subplot(grid[0, 4]),
    }
    draw_field_map(axes["map"], fields, theta)
    draw_extreme_h_bucket_distribution(axes["states"], fields, n_values, theta)
    draw_mean_field(axes["mean_field"], qsg_rows, n_values)
    draw_qsg_vote_shares(axes["votes"], qsg_rows, n_values)
    if crop_sets:
        draw_seed_field_mix(axes["qsg"], crop_sets)
    else:
        draw_qsg(axes["qsg"], qsg_rows)

    figure_paths = [
        FIGURE_DIR / f"{stem}.png",
        FIGURE_DIR / f"{stem}.pdf",
        FIGURE_DIR / f"{stem}.svg",
    ]
    for path in figure_paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return figure_paths


def parse_n_values(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-input",
        type=Path,
        action="append",
        default=[],
        help="CSV/JSONL repeated isolated crop-probe table. May be passed more than once.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Completed Flag Game run root for smoke testing from t=0 probes.",
    )
    parser.add_argument("--model", default=None, help="Model to show in the figure. Defaults to a large available group.")
    parser.add_argument("--target-country", default=None, help="Target country to show. Defaults to France if available.")
    parser.add_argument(
        "--rival-country",
        default=None,
        help="Optional rival country to use for h = log p(T)/p(R). Defaults to strongest isolated-probe rival.",
    )
    parser.add_argument("--out-stem", default=DEFAULT_STEM)
    parser.add_argument("--n-values", default=",".join(str(n) for n in DEFAULT_N_VALUES))
    parser.add_argument("--lambda-smoothing", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--theta", type=float, default=math.log(2.0))
    parser.add_argument("--qsg-trials", type=int, default=1000)
    parser.add_argument("--qsg-seed", type=int, default=20260502)
    parser.add_argument("--beta", type=float, default=1.30)
    parser.add_argument("--coupling", type=float, default=0.55)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--m-c", type=float, default=0.55)
    parser.add_argument("--q-c", type=float, default=0.45)
    parser.add_argument(
        "--figure-layout",
        choices=("full", "two_panel", "mechanism_story"),
        default="full",
        help=(
            "Draw the full diagnostic mechanism figure, a compressed two-panel candidate, "
            "or the four-panel measured-field mechanism story."
        ),
    )
    parser.add_argument(
        "--mechanism-run-root",
        type=Path,
        default=None,
        help=(
            "Optional N-scaling run root. When provided, Panels B-E use the measured h values "
            "for the actual crops assigned in condition/seed/N manifests."
        ),
    )
    parser.add_argument("--mechanism-condition", default="all_gpt_4o")
    parser.add_argument("--mechanism-seed", default="seed_0003")
    parser.add_argument(
        "--allow-missing-mechanism-crops",
        action="store_true",
        help="Draw a diagnostic figure even if some assigned crops do not have repeated-probe h estimates.",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--j-msg", type=float, default=1.10)
    parser.add_argument("--kappa", type=int, default=24)
    args = parser.parse_args()

    probe_rows: list[dict[str, Any]] = []
    source_parts: list[str] = []
    if args.probe_input:
        loaded = load_probe_inputs(args.probe_input)
        probe_rows.extend(loaded)
        source_parts.append(f"repeated probe inputs: {len(loaded)} rows")
    if args.run_root:
        loaded = load_run_root_initial_probes(args.run_root)
        probe_rows.extend(loaded)
        source_parts.append(f"run-root t0 probes: {len(loaded)} rows")
    if not probe_rows:
        raise SystemExit("Provide --probe-input and/or --run-root.")

    n_values = parse_n_values(args.n_values)
    fields = estimate_crop_fields(
        probe_rows,
        smoothing_lambda=args.lambda_smoothing,
        epsilon=args.epsilon,
        theta=args.theta,
        rival_country=args.rival_country,
    )
    if not fields:
        raise SystemExit("No crop fields could be estimated from the provided rows.")
    target_summary = summarize_target_models(fields, args.theta)
    selected = choose_default_selection(
        target_summary,
        requested_model=args.model,
        requested_target=args.target_country,
    )
    model = str(selected["model"])
    target = str(selected["target"])
    figure_fields = selected_fields(fields, model=model, target=target)
    coverage = coverage_rows(selected, n_values)
    crop_sets: list[dict[str, Any]] = []
    if args.mechanism_run_root:
        crop_sets = load_mechanism_crop_sets(
            figure_fields,
            run_root=args.mechanism_run_root,
            condition=args.mechanism_condition,
            seed=args.mechanism_seed,
            n_values=n_values,
            theta=args.theta,
            allow_missing=args.allow_missing_mechanism_crops,
        )
        qsg_rows = empirical_qsg_rows_from_crop_sets(
            crop_sets,
            trials=args.qsg_trials,
            seed=args.qsg_seed,
            beta=args.beta,
            coupling=args.coupling,
            eta=args.eta,
            alpha=args.alpha,
            j_msg=args.j_msg,
            kappa=args.kappa,
        )
    else:
        qsg_rows = empirical_qsg_rows(
            fields,
            model=model,
            target=target,
            n_values=n_values,
            trials=args.qsg_trials,
            seed=args.qsg_seed,
            beta=args.beta,
            coupling=args.coupling,
            eta=args.eta,
            m_c=args.m_c,
            q_c=args.q_c,
        )
    field_order_rows = (
        field_order_rows_from_crop_sets(crop_sets, beta=args.beta)
        if crop_sets
        else field_order_rows_from_pool(figure_fields, n_values=n_values, beta=args.beta)
    )
    endpoint_rows = load_endpoint_regime_summary(
        args.mechanism_run_root,
        condition=args.mechanism_condition,
        n_values=n_values,
    )
    figure_paths = draw_figure(
        stem=args.out_stem,
        fields=figure_fields,
        coverage=coverage,
        crop_sets=crop_sets,
        field_order_rows=field_order_rows,
        qsg_rows=qsg_rows,
        endpoint_rows=endpoint_rows,
        n_values=n_values,
        theta=args.theta,
        mechanism_seed=args.mechanism_seed if args.mechanism_run_root else None,
        figure_layout=args.figure_layout,
    )
    source_note = "; ".join(source_parts)
    output_paths = write_outputs(
        stem=args.out_stem,
        fields=fields,
        target_summary=target_summary,
        coverage=coverage,
        crop_sets=crop_sets,
        field_order_rows=field_order_rows,
        qsg_rows=qsg_rows,
        endpoint_rows=endpoint_rows,
        selected={
            **selected,
            "truth_positive_sign_convention": "h = log((p_T + epsilon)/(p_R + epsilon))",
            "mechanism_run_root": str(args.mechanism_run_root) if args.mechanism_run_root else None,
            "mechanism_condition": args.mechanism_condition,
            "mechanism_seed": args.mechanism_seed,
            "qsg_parameters": {
                "trials": args.qsg_trials,
                "seed": args.qsg_seed,
                "beta": args.beta,
                "coupling": args.coupling,
                "eta": args.eta,
                "m_c": args.m_c,
                "q_c": args.q_c,
                "alpha": args.alpha,
                "j_msg": args.j_msg,
                "kappa": args.kappa,
            },
        },
        probe_rows=probe_rows,
        source_note=source_note,
        figure_paths=figure_paths,
    )

    for path in [*output_paths.values(), *figure_paths]:
        print(f"Wrote {path}")
    if args.run_root and not args.probe_input:
        print(
            "Note: this run used existing t=0 probes as a smoke test. "
            "Use a repeated isolated crop-probe table for the paper estimate."
        )


if __name__ == "__main__":
    main()
