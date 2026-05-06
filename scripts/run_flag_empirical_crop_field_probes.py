#!/usr/bin/env python3
"""Collect repeated isolated crop probes for empirical Flag Game crop fields.

This is the data-collection companion for
paper/make_flag_game_empirical_crop_field_mechanism.py. It probes each crop
window B times with no social memory, writes one CSV row per response, and is
safe to resume because each row has a stable target/model/crop/rep key.

Example:
  python3 scripts/run_flag_empirical_crop_field_probes.py \
    --target France \
    --models gpt-4o \
    --B 50 \
    --country-pool stripe_expanded_24 \
    --tile-width 6 --tile-height 4 --render-scale 25 \
    --temperature 0.2 \
    --out results/flag_game/empirical_crop_field_probes/france_gpt4o_B50/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nnd.backends.parsing import ParseError
from nnd.flag_game.backend import build_backend
from nnd.flag_game.catalog import get_country_pool
from nnd.flag_game.crops import all_crop_boxes, crop_image, scale_crop_box
from nnd.flag_game.parsing import InteractionMessage
from nnd.flag_game.render import render_flag, save_png


DEFAULT_OUT = REPO_ROOT / "results" / "flag_game" / "empirical_crop_field_probes" / "results.csv"
FIELDNAMES = [
    "schema_version",
    "created_at_unix_ms",
    "target_country",
    "model",
    "backend",
    "country_pool",
    "allowed_countries_json",
    "B_requested",
    "rep",
    "task_key",
    "valid",
    "country",
    "clue",
    "reason",
    "error",
    "m",
    "temperature",
    "top_p",
    "max_tokens",
    "image_detail",
    "prompt_style",
    "canvas_width",
    "canvas_height",
    "tile_width",
    "tile_height",
    "render_scale",
    "crop_id",
    "crop_key",
    "crop_index",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
    "crop_pixel_top",
    "crop_pixel_left",
    "crop_pixel_height",
    "crop_pixel_width",
    "truth_flag_path",
    "crop_path",
    "prompt_social_susceptibility",
    "social_susceptibility",
]


def split_csv_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def read_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("task_key") or "").strip()
            if key:
                completed.add(key)
    return completed


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def task_key(
    *,
    target: str,
    model: str,
    country_pool: str,
    crop_index: int,
    top: int,
    left: int,
    height: int,
    width: int,
    rep: int,
    m: int,
) -> str:
    return (
        f"target={target}|model={model}|pool={country_pool}|"
        f"crop={crop_index}:{top},{left},{height},{width}|m={m}|rep={rep}"
    )


def crop_key(
    *,
    target: str,
    country_pool: str,
    top: int,
    left: int,
    height: int,
    width: int,
) -> str:
    return f"{country_pool}|{target}|{top}|{left}|{height}|{width}"


def parse_crop_indices(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    indices: set[int] = set()
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            indices.update(range(start, end + 1))
        else:
            indices.add(int(text))
    return indices


def crop_geometries_from_manifest(path: Path) -> set[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text())
    geometries: set[tuple[int, int, int, int]] = set()
    for assignment in payload.get("assignments", []):
        try:
            geometries.add(
                (
                    int(assignment["top"]),
                    int(assignment["left"]),
                    int(assignment["height"]),
                    int(assignment["width"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Could not read crop assignment geometry from {path}: {exc}") from exc
    if not geometries:
        raise SystemExit(f"No assignments found in crop manifest: {path}")
    return geometries


def materialize_target_assets(
    *,
    target: str,
    country_lookup: dict[str, Any],
    out_dir: Path,
    save_images: bool,
    canvas_width: int,
    canvas_height: int,
    render_scale: int,
) -> tuple[Any, Path]:
    flag = country_lookup[target]
    full_image = render_flag(
        flag,
        width=canvas_width * render_scale,
        height=canvas_height * render_scale,
    )
    truth_path = out_dir / "artifacts" / target.replace(" ", "_") / "truth_flag.png"
    if save_images:
        save_png(truth_path, full_image)
    return full_image, truth_path


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    targets = split_csv_items(args.target)
    if not targets:
        raise SystemExit("Pass at least one --target.")
    models = split_csv_items(args.models)
    if not models:
        raise SystemExit("Pass at least one --models value.")

    pool = get_country_pool(args.country_pool)
    country_lookup = {flag.country: flag for flag in pool}
    countries = [flag.country for flag in pool]
    missing = [target for target in targets if target not in country_lookup]
    if missing:
        raise SystemExit(
            f"Targets not in country_pool={args.country_pool}: {missing}. "
            f"Available examples: {countries[:8]}..."
        )

    boxes = all_crop_boxes(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )
    if args.crop_manifest:
        wanted_geometries: set[tuple[int, int, int, int]] = set()
        for manifest_path in args.crop_manifest:
            wanted_geometries.update(crop_geometries_from_manifest(manifest_path))
        available_geometries = {(box.top, box.left, box.height, box.width) for box in boxes}
        missing_geometries = sorted(wanted_geometries - available_geometries)
        if missing_geometries:
            preview = ", ".join(str(item) for item in missing_geometries[:6])
            raise SystemExit(
                f"{len(missing_geometries)} manifest crop geometries do not match the current crop grid. "
                f"First missing: {preview}"
            )
        boxes = [
            box
            for box in boxes
            if (box.top, box.left, box.height, box.width) in wanted_geometries
        ]
    selected_indices = parse_crop_indices(args.crop_indices)
    if selected_indices is not None:
        boxes = [box for box in boxes if box.crop_index in selected_indices]
    if args.crop_stride > 1:
        boxes = [box for idx, box in enumerate(boxes) if idx % args.crop_stride == 0]
    if args.shuffle_crops:
        rng = random.Random(args.seed)
        boxes = list(boxes)
        rng.shuffle(boxes)
    if args.crop_limit is not None:
        boxes = boxes[: args.crop_limit]
    if not boxes:
        raise SystemExit("No crop boxes selected.")

    tasks: list[dict[str, Any]] = []
    for target in targets:
        for model in models:
            for box in boxes:
                for rep in range(args.B):
                    key = task_key(
                        target=target,
                        model=model,
                        country_pool=args.country_pool,
                        crop_index=box.crop_index,
                        top=box.top,
                        left=box.left,
                        height=box.height,
                        width=box.width,
                        rep=rep,
                        m=args.m,
                    )
                    tasks.append(
                        {
                            "target": target,
                            "model": model,
                            "box": box,
                            "rep": rep,
                            "task_key": key,
                            "countries": countries,
                        }
                    )
    return tasks


def backend_for_model(
    *,
    args: argparse.Namespace,
    model: str,
    country_lookup: dict[str, Any],
    debug_dir: Path,
) -> Any:
    return build_backend(
        backend_name=args.backend,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        debug_dir=debug_dir / model.replace("/", "_"),
        image_detail=args.image_detail,
        seed=args.seed,
        social_susceptibility=args.social_susceptibility,
        prompt_social_susceptibility=args.prompt_social_susceptibility,
        prompt_style=args.prompt_style,
        country_lookup=country_lookup,
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", default=None, help="Target country. Repeat or comma-separate.")
    parser.add_argument("--models", action="append", default=None, help="Model name(s). Repeat or comma-separate.")
    parser.add_argument("--B", type=int, default=50, help="Repeated isolated probes per selected crop.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=("openai", "anthropic", "scripted"), default="openai")
    parser.add_argument("--country-pool", default="stripe_expanded_24")
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--image-detail", choices=("auto", "low", "high", "original"), default="high")
    parser.add_argument("--m", type=int, choices=(1, 3), default=1, help="Isolated response schema.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--prompt-style", choices=("closed_country_list", "open_country"), default="closed_country_list")
    parser.add_argument("--prompt-social-susceptibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--social-susceptibility", type=float, default=0.5)
    parser.add_argument("--crop-indices", default=None, help="Comma/range list, e.g. 0,3,10-20.")
    parser.add_argument(
        "--crop-manifest",
        type=Path,
        action="append",
        default=[],
        help="Trial manifest whose assigned crop geometries should be probed. May be repeated.",
    )
    parser.add_argument("--crop-stride", type=int, default=1, help="Use every kth crop after crop-index filtering.")
    parser.add_argument("--crop-limit", type=int, default=None, help="Pilot limit after filtering/shuffling.")
    parser.add_argument("--shuffle-crops", action="store_true")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls.")
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing task keys in --out.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.B < 1:
        raise SystemExit("--B must be >= 1.")
    if args.crop_stride < 1:
        raise SystemExit("--crop-stride must be >= 1.")

    tasks = build_tasks(args)
    completed = set() if args.overwrite else read_completed_keys(args.out)
    planned = [task for task in tasks if task["task_key"] not in completed]

    out_dir = args.out.parent
    targets = split_csv_items(args.target)
    models = split_csv_items(args.models)
    manifest_path = out_dir / "manifest.json"

    print(f"Output CSV: {args.out}")
    print(f"Manifest: {manifest_path}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(tasks) - len(planned)}")
    print(f"Planned calls: {len(planned)}")
    if args.dry_run:
        for task in planned[:10]:
            box = task["box"]
            print(
                f"- {task['target']} {task['model']} crop={box.crop_index} "
                f"top={box.top} left={box.left} rep={task['rep']}"
            )
        if len(planned) > 10:
            print(f"... {len(planned) - 10} more")
        return

    country_lookup = {flag.country: flag for flag in get_country_pool(args.country_pool)}
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "script": str(Path(__file__).resolve()),
            "out": str(args.out),
            "backend": args.backend,
            "models": models,
            "targets": targets,
            "B": args.B,
            "country_pool": args.country_pool,
            "allowed_countries": [flag.country for flag in get_country_pool(args.country_pool)],
            "canvas_width": args.canvas_width,
            "canvas_height": args.canvas_height,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "render_scale": args.render_scale,
            "image_detail": args.image_detail,
            "m": args.m,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "prompt_style": args.prompt_style,
            "prompt_social_susceptibility": args.prompt_social_susceptibility,
            "social_susceptibility": args.social_susceptibility,
            "crop_manifests": [str(path) for path in args.crop_manifest],
            "total_tasks": len(tasks),
            "completed_existing": len(completed),
            "planned_tasks": len(planned),
            "created_at_unix_ms": int(time.time() * 1000),
        },
    )

    target_images: dict[str, tuple[Any, Path]] = {}
    for target in targets:
        target_images[target] = materialize_target_assets(
            target=target,
            country_lookup=country_lookup,
            out_dir=out_dir,
            save_images=args.save_images,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            render_scale=args.render_scale,
        )

    backends: dict[str, Any] = {}
    prepared_crops: dict[tuple[str, str], str] = {}
    countries = [flag.country for flag in get_country_pool(args.country_pool)]
    allowed_json = json.dumps(countries, ensure_ascii=True)
    debug_dir = out_dir / "debug"

    for index, task in enumerate(planned, start=1):
        target = str(task["target"])
        model = str(task["model"])
        box = task["box"]
        rep = int(task["rep"])
        if model not in backends:
            backends[model] = backend_for_model(
                args=args,
                model=model,
                country_lookup=country_lookup,
                debug_dir=debug_dir,
            )
        backend = backends[model]
        full_image, truth_path = target_images[target]
        scaled_box = scale_crop_box(box, args.render_scale)
        crop = crop_image(full_image, scaled_box)
        crop_path = (
            out_dir
            / "artifacts"
            / target.replace(" ", "_")
            / f"crop_{box.crop_index:04d}_top{box.top}_left{box.left}.png"
        )
        if args.save_images and not crop_path.exists():
            save_png(crop_path, crop)
        prepared_key = (model, str(crop_path if args.save_images else task["task_key"].rsplit("|rep=", 1)[0]))
        if prepared_key not in prepared_crops:
            prepared_crops[prepared_key] = backend.prepare_crop(crop)

        valid = False
        country = ""
        clue = ""
        reason = ""
        error = ""
        try:
            message = backend.probe(
                countries=countries,
                prepared_crop=prepared_crops[prepared_key],
                memory_lines=[],
                m=args.m,
            )
            if isinstance(message, str):
                message = InteractionMessage(country=message)
            valid = True
            country = message.country
            clue = message.clue or ""
            reason = message.reason or ""
        except ParseError as exc:
            error = str(exc)

        row = {
            "schema_version": 1,
            "created_at_unix_ms": int(time.time() * 1000),
            "target_country": target,
            "model": model,
            "backend": args.backend,
            "country_pool": args.country_pool,
            "allowed_countries_json": allowed_json,
            "B_requested": args.B,
            "rep": rep,
            "task_key": task["task_key"],
            "valid": valid,
            "country": country,
            "clue": clue,
            "reason": reason,
            "error": error,
            "m": args.m,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "image_detail": args.image_detail,
            "prompt_style": args.prompt_style,
            "canvas_width": args.canvas_width,
            "canvas_height": args.canvas_height,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "render_scale": args.render_scale,
            "crop_id": f"{target}|{box.crop_index}",
            "crop_key": crop_key(
                target=target,
                country_pool=args.country_pool,
                top=box.top,
                left=box.left,
                height=box.height,
                width=box.width,
            ),
            "crop_index": box.crop_index,
            "crop_top": box.top,
            "crop_left": box.left,
            "crop_height": box.height,
            "crop_width": box.width,
            "crop_pixel_top": scaled_box.top,
            "crop_pixel_left": scaled_box.left,
            "crop_pixel_height": scaled_box.height,
            "crop_pixel_width": scaled_box.width,
            "truth_flag_path": str(truth_path) if args.save_images else "",
            "crop_path": str(crop_path) if args.save_images else "",
            "prompt_social_susceptibility": args.prompt_social_susceptibility,
            "social_susceptibility": args.social_susceptibility,
        }
        append_row(args.out, row)

        if index == 1 or index % 25 == 0 or index == len(planned):
            print(
                f"[{index}/{len(planned)}] {target} {model} "
                f"crop={box.crop_index} rep={rep} valid={valid} country={row['country']}"
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    usage_path = out_dir / "usage_summary.json"
    usage = {
        model: backend.usage_summary()
        for model, backend in backends.items()
        if hasattr(backend, "usage_summary")
    }
    write_json(usage_path, usage)
    print(f"Wrote usage: {usage_path}")


if __name__ == "__main__":
    run()
