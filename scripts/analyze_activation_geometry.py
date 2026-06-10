#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze representation geometry from captured flag-game activations."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory, batch directory, or multiple directories containing activations/index.jsonl.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--feature",
        choices=("last_prompt_token", "mean_prompt"),
        default="last_prompt_token",
    )
    parser.add_argument(
        "--call-type",
        choices=("probe", "interaction", "all"),
        default="probe",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer ids to include. Defaults to every captured layer.",
    )
    parser.add_argument(
        "--t",
        type=int,
        nargs="*",
        default=None,
        help="Optional timestep filter. Defaults to every captured timestep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.runs)
    if not run_dirs:
        raise SystemExit("No run directories with activations/index.jsonl were found.")

    samples = load_samples(run_dirs, args)
    if not samples:
        raise SystemExit("No activation samples matched the requested filters.")

    out_dir = args.out or default_out_dir(run_dirs, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = sample_table(samples)
    pair_rows = pairwise_agent_cosines(samples)
    drift_rows = temporal_drift(samples)
    layer_summary_rows = summarize_pairs(pair_rows)
    drift_summary_rows = summarize_drift(drift_rows)

    pd.DataFrame(sample_rows).to_csv(out_dir / "activation_samples.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "agent_pair_cosine.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(out_dir / "agent_temporal_drift.csv", index=False)
    pd.DataFrame(layer_summary_rows).to_csv(out_dir / "by_layer_similarity_summary.csv", index=False)
    pd.DataFrame(drift_summary_rows).to_csv(out_dir / "by_layer_temporal_summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "feature": args.feature,
                "call_type": args.call_type,
                "run_dirs": [str(path) for path in run_dirs],
                "sample_count": len(samples),
                "pair_count": len(pair_rows),
                "temporal_drift_count": len(drift_rows),
                "outputs": [
                    "activation_samples.csv",
                    "agent_pair_cosine.csv",
                    "agent_temporal_drift.csv",
                    "by_layer_similarity_summary.csv",
                    "by_layer_temporal_summary.csv",
                ],
            },
            handle,
            indent=2,
        )
    print(f"Wrote activation-geometry analysis to {out_dir}")
    print(f"Samples: {len(samples)} | same-timestep agent pairs: {len(pair_rows)} | temporal comparisons: {len(drift_rows)}")


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if (resolved / "activations" / "index.jsonl").exists():
            candidates = [resolved]
        else:
            candidates = [index.parent.parent for index in resolved.glob("**/activations/index.jsonl")]
        for candidate in sorted(candidates):
            if candidate not in seen:
                seen.add(candidate)
                run_dirs.append(candidate)
    return run_dirs


def default_out_dir(run_dirs: list[Path], args: argparse.Namespace) -> Path:
    name = f"activation_geometry_{args.feature}"
    if args.call_type != "all":
        name = f"{name}_{args.call_type}"
    if len(run_dirs) == 1:
        return run_dirs[0] / name
    return run_dirs[0].parent / name


def load_samples(run_dirs: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    wanted_layers = set(args.layers) if args.layers is not None else None
    wanted_timesteps = set(args.t) if args.t is not None else None
    for run_dir in run_dirs:
        probe_rows = {
            (int(row.get("t", -1)), int(row.get("agent_id", -1))): row
            for row in read_jsonl(run_dir / "probes.jsonl")
        }
        interaction_rows = {
            (int(row.get("t", -1)), int(row.get("speaker_id", -1))): row
            for row in read_jsonl(run_dir / "interactions.jsonl")
        }
        manifest = read_json(run_dir / "trial_manifest.json")
        assignments = {
            int(row["agent_id"]): row
            for row in manifest.get("assignments", [])
            if "agent_id" in row
        }

        for row in read_jsonl(run_dir / "activations" / "index.jsonl"):
            call_type = str(row.get("call_type", ""))
            if args.call_type != "all" and call_type != args.call_type:
                continue
            t_value = int(row.get("t", -1))
            if wanted_timesteps is not None and t_value not in wanted_timesteps:
                continue
            agent_id = int(row.get("agent_id", row.get("speaker_id", -1)))
            payload = load_tensor_payload(resolve_tensor_path(run_dir, row))
            if args.feature not in payload:
                raise KeyError(f"{row.get('tensor_path')} does not contain feature {args.feature!r}")
            layers = [int(value) for value in payload["layers"].detach().cpu().numpy().tolist()]
            features = payload[args.feature].detach().float().cpu().numpy()
            if wanted_layers is not None:
                keep = [idx for idx, layer in enumerate(layers) if layer in wanted_layers]
                layers = [layers[idx] for idx in keep]
                features = features[keep, :]
            if not layers:
                continue
            behavior_row = behavior_for_call(
                call_type=call_type,
                t=t_value,
                agent_id=agent_id,
                probe_rows=probe_rows,
                interaction_rows=interaction_rows,
            )
            samples.append(
                {
                    "run_dir": run_dir,
                    "run_id": run_dir.name,
                    "call_id": row.get("call_id"),
                    "call_type": call_type,
                    "t": t_value,
                    "agent_id": agent_id,
                    "m": row.get("m"),
                    "model": row.get("model"),
                    "truth_country": row.get("truth_country"),
                    "predicted_country": behavior_row.get("country") if behavior_row else None,
                    "correct": behavior_row.get("correct") if behavior_row else None,
                    "crop_diagnostic": row.get("crop_diagnostic"),
                    "crop_box": row.get("crop_box") or assignments.get(agent_id),
                    "layers": layers,
                    "features": features,
                    "metadata": row,
                }
            )
    return samples


def behavior_for_call(
    *,
    call_type: str,
    t: int,
    agent_id: int,
    probe_rows: dict[tuple[int, int], dict[str, Any]],
    interaction_rows: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any] | None:
    if call_type == "probe":
        return probe_rows.get((t, agent_id))
    if call_type == "interaction":
        return interaction_rows.get((t, agent_id))
    return None


def sample_table(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        crop_diagnostic = sample.get("crop_diagnostic") or {}
        crop_box = sample.get("crop_box") or {}
        rows.append(
            {
                "run_dir": str(sample["run_dir"]),
                "run_id": sample["run_id"],
                "call_id": sample["call_id"],
                "call_type": sample["call_type"],
                "t": sample["t"],
                "agent_id": sample["agent_id"],
                "m": sample["m"],
                "model": sample["model"],
                "truth_country": sample["truth_country"],
                "predicted_country": sample["predicted_country"],
                "correct": sample["correct"],
                "layer_count": len(sample["layers"]),
                "layers": json.dumps(sample["layers"]),
                "crop_top": crop_box.get("top"),
                "crop_left": crop_box.get("left"),
                "crop_height": crop_box.get("height"),
                "crop_width": crop_box.get("width"),
                "truth_compatible": crop_diagnostic.get("truth_compatible"),
                "informativeness_label": crop_diagnostic.get("informativeness_label"),
                "compatible_country_count": crop_diagnostic.get("compatible_country_count"),
                "is_unique": crop_diagnostic.get("is_unique"),
            }
        )
    return rows


def pairwise_agent_cosines(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Path, str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["run_dir"], sample["call_type"], int(sample["t"]))].append(sample)

    rows: list[dict[str, Any]] = []
    for (run_dir, call_type, t_value), group in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])):
        group = sorted(group, key=lambda sample: int(sample["agent_id"]))
        for left_idx, left in enumerate(group):
            for right in group[left_idx + 1 :]:
                shared_layers = sorted(set(left["layers"]).intersection(right["layers"]))
                left_lookup = layer_feature_lookup(left)
                right_lookup = layer_feature_lookup(right)
                for layer in shared_layers:
                    left_vec = left_lookup[layer]
                    right_vec = right_lookup[layer]
                    rows.append(
                        {
                            "run_dir": str(run_dir),
                            "run_id": left["run_id"],
                            "call_type": call_type,
                            "t": t_value,
                            "layer": layer,
                            "agent_a": left["agent_id"],
                            "agent_b": right["agent_id"],
                            "cosine_similarity": cosine(left_vec, right_vec),
                            "l2_distance": l2_distance(left_vec, right_vec),
                            "truth_country": left["truth_country"],
                            "predicted_country_a": left["predicted_country"],
                            "predicted_country_b": right["predicted_country"],
                            "same_predicted_country": same_or_none(left["predicted_country"], right["predicted_country"]),
                            "correct_a": left["correct"],
                            "correct_b": right["correct"],
                        }
                    )
    return rows


def temporal_drift(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Path, str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["run_dir"], sample["call_type"], int(sample["agent_id"]))].append(sample)

    rows: list[dict[str, Any]] = []
    for (run_dir, call_type, agent_id), group in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])):
        group = sorted(group, key=lambda sample: int(sample["t"]))
        if not group:
            continue
        baseline = group[0]
        previous = None
        baseline_lookup = layer_feature_lookup(baseline)
        for sample in group:
            sample_lookup = layer_feature_lookup(sample)
            previous_lookup = layer_feature_lookup(previous) if previous is not None else None
            for layer in sorted(set(baseline_lookup).intersection(sample_lookup)):
                current_vec = sample_lookup[layer]
                base_vec = baseline_lookup[layer]
                row = {
                    "run_dir": str(run_dir),
                    "run_id": sample["run_id"],
                    "call_type": call_type,
                    "agent_id": agent_id,
                    "t": sample["t"],
                    "baseline_t": baseline["t"],
                    "previous_t": previous["t"] if previous is not None else None,
                    "layer": layer,
                    "cosine_to_initial": cosine(current_vec, base_vec),
                    "l2_to_initial": l2_distance(current_vec, base_vec),
                    "truth_country": sample["truth_country"],
                    "predicted_country": sample["predicted_country"],
                    "correct": sample["correct"],
                }
                if previous_lookup is not None and layer in previous_lookup:
                    prev_vec = previous_lookup[layer]
                    row["cosine_to_previous"] = cosine(current_vec, prev_vec)
                    row["l2_to_previous"] = l2_distance(current_vec, prev_vec)
                else:
                    row["cosine_to_previous"] = None
                    row["l2_to_previous"] = None
                rows.append(row)
            previous = sample
    return rows


def summarize_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pair_rows:
        return []
    df = pd.DataFrame(pair_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["call_type", "t", "layer"], dropna=False):
        call_type, t_value, layer = keys
        rows.append(
            {
                "call_type": call_type,
                "t": t_value,
                "layer": layer,
                "pair_count": len(group),
                "mean_cosine_similarity": float(group["cosine_similarity"].mean()),
                "std_cosine_similarity": float(group["cosine_similarity"].std(ddof=0)),
                "min_cosine_similarity": float(group["cosine_similarity"].min()),
                "max_cosine_similarity": float(group["cosine_similarity"].max()),
                "mean_l2_distance": float(group["l2_distance"].mean()),
            }
        )
    return rows


def summarize_drift(drift_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not drift_rows:
        return []
    df = pd.DataFrame(drift_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["call_type", "t", "layer"], dropna=False):
        call_type, t_value, layer = keys
        rows.append(
            {
                "call_type": call_type,
                "t": t_value,
                "layer": layer,
                "sample_count": len(group),
                "mean_cosine_to_initial": float(group["cosine_to_initial"].mean()),
                "std_cosine_to_initial": float(group["cosine_to_initial"].std(ddof=0)),
                "mean_l2_to_initial": float(group["l2_to_initial"].mean()),
                "mean_cosine_to_previous": none_if_nan(group["cosine_to_previous"].mean()),
                "mean_l2_to_previous": none_if_nan(group["l2_to_previous"].mean()),
            }
        )
    return rows


def layer_feature_lookup(sample: dict[str, Any] | None) -> dict[int, np.ndarray]:
    if sample is None:
        return {}
    return {
        int(layer): sample["features"][idx]
        for idx, layer in enumerate(sample["layers"])
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def same_or_none(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return left == right


def none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        return value
    return float(value)


def resolve_tensor_path(run_dir: Path, row: dict[str, Any]) -> Path:
    tensor_path = Path(str(row["tensor_path"]))
    if tensor_path.is_absolute():
        return tensor_path
    return run_dir / "activations" / tensor_path


def load_tensor_payload(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
