#!/usr/bin/env python3
"""Run a paired visual-only Flag Game crop audit.

This script creates the CSV consumed by paper/make_flag_visual_only_audit_visuals.py.
It samples matched flag crops, asks each requested model to identify the country
from the same crop-only prompt, and records both country choices and visual
reasons for qualitative side-by-side examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from nnd.backends.parsing import ParseError
from nnd.flag_game.backend import build_backend
from nnd.flag_game.catalog import FlagSpec, get_country_pool
from nnd.flag_game.crops import CropBox, all_crop_boxes, crop_image, scale_crop_box
from nnd.flag_game.diagnostics import build_crop_compatibility_cache, describe_crop_informativeness
from nnd.flag_game.parsing import InteractionMessage
from nnd.flag_game.render import render_flag, save_png


DEFAULT_OUT = ROOT / "results" / "flag_game" / "visual_only_paired_crop_audit"
DEFAULT_MODELS = ["gpt-4o", "gpt-5.4"]
DEFAULT_TRUTHS = ["Czech Republic", "Peru", "Guinea", "Bahamas"]
INFO_ORDER = ["unique", "narrow", "moderate", "ambiguous"]


@dataclass(frozen=True)
class Stimulus:
    pair_id: str
    truth_country: str
    crop_box: CropBox
    crop_image: np.ndarray
    diagnostic: dict[str, Any]
    crop_path: Path
    allowed_countries: tuple[str, ...]
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a paired visual-only crop audit for GPT-4o versus GPT-5.4 "
            "and optionally render the paper diagnostics."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help=(
            "Figure output directory used with --make-figures. Defaults to paper/exports/figures "
            "for the standard audit path, otherwise <out>/figures."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Summary output directory used with --make-figures. Defaults to paper/exports/data "
            "for the standard audit path, otherwise <out>/data."
        ),
    )
    parser.add_argument("--backend", default=None, choices=["openai", "anthropic", "scripted"])
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--country-pool", default="stripe_plus_real_triangle_28")
    parser.add_argument(
        "--all-truth-countries",
        action="store_true",
        help="Use every country in the selected country pool as a truth country.",
    )
    parser.add_argument("--truth-country", action="append", dest="truth_countries", default=None)
    parser.add_argument(
        "--per-label",
        type=int,
        default=3,
        help="Number of truth-compatible crops to sample per truth country and informativeness label.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Optional cap on sampled crop pairs, useful for a cheap pilot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--image-detail", default="high", choices=["auto", "low", "high", "original"])
    parser.add_argument(
        "--m",
        type=int,
        default=3,
        choices=[1, 3],
        help="Use m=3 to collect one-sentence visual reasons for examples.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write plan/crops and exit before API calls.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild results.csv from existing JSONL only.")
    parser.add_argument("--force", action="store_true", help="Rerun model/crop rows already in results.jsonl.")
    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Run paper/make_flag_visual_only_audit_visuals.py after results.csv is written.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=12,
        help="Number of qualitative failure candidates to include in the gallery when --make-figures is used.",
    )
    parser.add_argument(
        "--selected-example-pair-id",
        action="append",
        dest="selected_example_pair_ids",
        default=None,
        help=(
            "Pair ID to force into the three-example qualitative figure when "
            "--make-figures is used. Repeat to set the figure order."
        ),
    )
    parser.add_argument(
        "--manual-adjudication",
        type=Path,
        default=None,
        help=(
            "Filled manual adjudication CSV to pass to the figure script. "
            "Use the exported manual_adjudication_template columns."
        ),
    )
    return parser.parse_args()


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "item"


def country_names(pool: list[FlagSpec]) -> list[str]:
    return [flag.country for flag in pool]


def country_lookup(pool: list[FlagSpec]) -> dict[str, FlagSpec]:
    return {flag.country: flag for flag in pool}


def stimulus_pair_id(truth_country: str, box: CropBox) -> str:
    return (
        f"{safe_slug(truth_country)}"
        f"__idx{box.crop_index:03d}"
        f"__t{box.top:02d}_l{box.left:02d}_h{box.height:02d}_w{box.width:02d}"
    )


def stable_int_seed(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def build_stimuli(args: argparse.Namespace) -> list[Stimulus]:
    if args.per_label < 1:
        raise ValueError("--per-label must be >= 1")

    rng = random.Random(args.seed)
    pool = get_country_pool(args.country_pool)
    countries = country_names(pool)
    lookup = country_lookup(pool)
    truths = list(countries) if args.all_truth_countries else args.truth_countries or list(DEFAULT_TRUTHS)
    missing = [country for country in truths if country not in lookup]
    if missing:
        raise ValueError(f"Truth countries not in pool {args.country_pool}: {missing}")

    compatibility_cache = build_crop_compatibility_cache(
        pool,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        render_scale=args.render_scale,
    )

    out: list[Stimulus] = []
    for truth_country in truths:
        truth_flag = lookup[truth_country]
        full_image = render_flag(
            truth_flag,
            width=args.canvas_width * args.render_scale,
            height=args.canvas_height * args.render_scale,
        )
        by_label: dict[str, list[tuple[CropBox, np.ndarray, dict[str, Any]]]] = {
            label: [] for label in INFO_ORDER
        }
        for box in all_crop_boxes(
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        ):
            scaled_box = scale_crop_box(box, args.render_scale)
            image = crop_image(full_image, scaled_box)
            diagnostic = describe_crop_informativeness(
                image,
                country_order=countries,
                compatibility_cache=compatibility_cache,
            )
            if truth_country not in diagnostic["compatible_countries"]:
                continue
            label = diagnostic["informativeness_label"]
            if label in by_label:
                by_label[label].append((box, image, diagnostic))

        for label in INFO_ORDER:
            candidates = list(by_label[label])
            if not candidates:
                continue
            rng.shuffle(candidates)
            selected = sorted(
                candidates[: args.per_label],
                key=lambda item: (item[0].crop_index, item[0].top, item[0].left),
            )
            for box, image, diagnostic in selected:
                pair_id = stimulus_pair_id(truth_country, box)
                crop_path = args.out / "artifacts" / f"{pair_id}.png"
                allowed = list(countries)
                random.Random(f"{args.seed}:{pair_id}:allowed").shuffle(allowed)
                out.append(
                    Stimulus(
                        pair_id=pair_id,
                        truth_country=truth_country,
                        crop_box=box,
                        crop_image=image,
                        diagnostic=diagnostic,
                        crop_path=crop_path,
                        allowed_countries=tuple(allowed),
                        seed=stable_int_seed(f"{args.seed}:{pair_id}"),
                    )
                )

    out = sorted(out, key=lambda item: (item.truth_country, item.pair_id))
    if args.limit_pairs is not None:
        out = out[: args.limit_pairs]
    return out


def write_plan(args: argparse.Namespace, stimuli: list[Stimulus]) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        save_png(stimulus.crop_path, stimulus.crop_image)
        rows.append(
            {
                "pair_id": stimulus.pair_id,
                "truth_country": stimulus.truth_country,
                "crop_path": str(stimulus.crop_path),
                **{f"crop_{key}": value for key, value in stimulus.crop_box.to_dict().items()},
                "crop_compatible_country_count": stimulus.diagnostic["compatible_country_count"],
                "crop_compatible_countries": json.dumps(stimulus.diagnostic["compatible_countries"]),
                "crop_informativeness_bits": stimulus.diagnostic["informativeness_bits"],
                "crop_informativeness_label": stimulus.diagnostic["informativeness_label"],
                "allowed_countries": json.dumps(stimulus.allowed_countries, ensure_ascii=True),
                "seed": stimulus.seed,
            }
        )
    plan_path = args.out / "stimulus_plan.csv"
    with plan_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["pair_id"])
        writer.writeheader()
        writer.writerows(rows)
    return plan_path


def existing_response_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair_id = row.get("pair_id")
            model = row.get("model")
            if isinstance(pair_id, str) and isinstance(model, str):
                keys.add((pair_id, model))
    return keys


def row_from_response(
    *,
    args: argparse.Namespace,
    stimulus: Stimulus,
    model: str,
    backend_name: str,
    message: InteractionMessage | None,
    valid: bool,
    error: str | None,
) -> dict[str, Any]:
    choice = message.country if message is not None else None
    reason = message.reason if message is not None else None
    compatible = set(stimulus.diagnostic["compatible_countries"])
    return {
        "pair_id": stimulus.pair_id,
        "model": model,
        "m": args.m,
        "backend": backend_name,
        "truth_country": stimulus.truth_country,
        "choice_country": choice,
        "reason": reason,
        "valid": valid,
        "correct": bool(valid and choice == stimulus.truth_country),
        "choice_crop_compatible": bool(valid and choice in compatible),
        "error": error,
        "crop_path": str(stimulus.crop_path),
        **{f"crop_{key}": value for key, value in stimulus.crop_box.to_dict().items()},
        "crop_compatible_country_count": stimulus.diagnostic["compatible_country_count"],
        "crop_compatible_countries": json.dumps(stimulus.diagnostic["compatible_countries"]),
        "crop_informativeness_bits": stimulus.diagnostic["informativeness_bits"],
        "crop_informativeness_label": stimulus.diagnostic["informativeness_label"],
        "allowed_countries": json.dumps(stimulus.allowed_countries, ensure_ascii=True),
        "country_pool": args.country_pool,
        "image_detail": args.image_detail,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": stimulus.seed,
    }


def run_audit(args: argparse.Namespace, stimuli: list[Stimulus]) -> None:
    models = args.models or list(DEFAULT_MODELS)
    backend_name = args.backend or "openai"
    pool = get_country_pool(args.country_pool)
    lookup = country_lookup(pool)
    results_path = args.out / "results.jsonl"
    if args.force and results_path.exists():
        results_path.write_text("")
    completed = set() if args.force else existing_response_keys(results_path)
    backend_cache: dict[str, Any] = {}
    prepared_cache: dict[tuple[str, str], Any] = {}

    def backend_for(model: str) -> Any:
        if model not in backend_cache:
            backend_cache[model] = build_backend(
                backend_name=backend_name,
                model=model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                debug_dir=args.out / "debug" / safe_slug(model),
                image_detail=args.image_detail,
                seed=args.seed,
                social_susceptibility=0.0,
                prompt_social_susceptibility=False,
                prompt_style="closed_country_list",
                country_lookup=lookup,
            )
        return backend_cache[model]

    def prepared_crop(model: str, stimulus: Stimulus) -> Any:
        key = (model, stimulus.pair_id)
        if key not in prepared_cache:
            prepared_cache[key] = backend_for(model).prepare_crop(stimulus.crop_image)
        return prepared_cache[key]

    total = len(stimuli) * len(models)
    pending = [
        (stimulus, model)
        for stimulus in stimuli
        for model in models
        if (stimulus.pair_id, model) not in completed
    ]
    print(
        f"Visual-only audit: {len(stimuli)} crop pairs, {len(models)} models, "
        f"{total} model responses total, {len(pending)} pending. Backend={backend_name}."
    )

    if not pending:
        return

    with results_path.open("a") as handle:
        for index, (stimulus, model) in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] {model} on {stimulus.pair_id} "
                f"({stimulus.diagnostic['informativeness_label']})"
            )
            try:
                message = backend_for(model).probe(
                    countries=list(stimulus.allowed_countries),
                    prepared_crop=prepared_crop(model, stimulus),
                    memory_lines=[],
                    m=args.m,
                )
                row = row_from_response(
                    args=args,
                    stimulus=stimulus,
                    model=model,
                    backend_name=backend_name,
                    message=message,
                    valid=True,
                    error=None,
                )
            except (ParseError, Exception) as exc:
                row = row_from_response(
                    args=args,
                    stimulus=stimulus,
                    model=model,
                    backend_name=backend_name,
                    message=None,
                    valid=False,
                    error=repr(exc),
                )
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()

    usage = {
        model: backend.usage_summary()
        for model, backend in backend_cache.items()
        if hasattr(backend, "usage_summary")
    }
    (args.out / "api_usage_summary.json").write_text(json.dumps(usage, indent=2, sort_keys=True))


def write_results_csv(out_dir: Path) -> Path:
    results_path = out_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    csv_path = out_dir / "results.csv"
    if not rows:
        csv_path.write_text("")
        return csv_path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_run_config(args: argparse.Namespace, stimuli: list[Stimulus]) -> None:
    planned_truths = sorted({stimulus.truth_country for stimulus in stimuli})
    config = {
        "out": str(args.out),
        "backend": args.backend or "openai",
        "models": args.models or list(DEFAULT_MODELS),
        "country_pool": args.country_pool,
        "all_truth_countries": args.all_truth_countries,
        "truth_countries": planned_truths,
        "per_label": args.per_label,
        "limit_pairs": args.limit_pairs,
        "seed": args.seed,
        "canvas_width": args.canvas_width,
        "canvas_height": args.canvas_height,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "render_scale": args.render_scale,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "image_detail": args.image_detail,
        "m": args.m,
        "n_stimuli": len(stimuli),
    }
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def figure_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.figure_dir is not None and args.data_dir is not None:
        return args.figure_dir, args.data_dir

    is_standard_out = args.out.resolve() == DEFAULT_OUT.resolve()
    figure_dir = args.figure_dir
    data_dir = args.data_dir
    if figure_dir is None:
        figure_dir = ROOT / "paper" / "exports" / "figures" if is_standard_out else args.out / "figures"
    if data_dir is None:
        data_dir = ROOT / "paper" / "exports" / "data" if is_standard_out else args.out / "data"
    return figure_dir, data_dir


def run_figure_script(
    results_csv: Path,
    *,
    figure_dir: Path,
    data_dir: Path,
    max_examples: int,
    selected_example_pair_ids: list[str] | None,
    manual_adjudication: Path | None,
) -> None:
    script = ROOT / "paper" / "make_flag_visual_only_audit_visuals.py"
    command = [
        sys.executable,
        str(script),
        "--source",
        str(results_csv),
        "--figure-dir",
        str(figure_dir),
        "--data-dir",
        str(data_dir),
        "--max-examples",
        str(max_examples),
    ]
    for pair_id in selected_example_pair_ids or []:
        command.extend(["--selected-example-pair-id", pair_id])
    if manual_adjudication is not None:
        command.extend(["--manual-adjudication", str(manual_adjudication)])
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.report_only:
        results_csv = write_results_csv(args.out)
        print(f"Wrote audit CSV: {results_csv}")
        if args.make_figures:
            figure_dir, data_dir = figure_output_dirs(args)
            run_figure_script(
                results_csv,
                figure_dir=figure_dir,
                data_dir=data_dir,
                max_examples=args.max_examples,
                selected_example_pair_ids=args.selected_example_pair_ids,
                manual_adjudication=args.manual_adjudication,
            )
        return

    stimuli = build_stimuli(args)
    if not stimuli:
        raise RuntimeError("No stimuli were sampled. Try different truth countries or crop geometry.")

    plan_path = write_plan(args, stimuli)
    write_run_config(args, stimuli)
    print(f"Wrote stimulus plan: {plan_path}")
    print(
        f"Planned {len(stimuli)} crop pairs and "
        f"{len(stimuli) * len(args.models or DEFAULT_MODELS)} model responses."
    )

    if args.dry_run:
        print("Dry run complete; no model calls made.")
        return

    run_audit(args, stimuli)
    results_csv = write_results_csv(args.out)
    print(f"Wrote audit CSV: {results_csv}")

    if args.make_figures:
        figure_dir, data_dir = figure_output_dirs(args)
        run_figure_script(
            results_csv,
            figure_dir=figure_dir,
            data_dir=data_dir,
            max_examples=args.max_examples,
            selected_example_pair_ids=args.selected_example_pair_ids,
            manual_adjudication=args.manual_adjudication,
        )


if __name__ == "__main__":
    main()
