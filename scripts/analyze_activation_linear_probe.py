#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


TARGETS = (
    "truth_country",
    "predicted_country",
    "correct",
    "truth_compatible",
    "is_unique",
    "informativeness_label",
    "compatible_country_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train simple per-layer linear probes over captured flag-game activations."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory, batch directory, or multiple directories containing activations/index.jsonl.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target", choices=TARGETS, default="truth_country")
    parser.add_argument(
        "--feature",
        choices=("last_prompt_token", "mean_prompt"),
        default="last_prompt_token",
    )
    parser.add_argument("--call-type", default="probe")
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty.")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer ids to evaluate. Defaults to every captured layer.",
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

    out_dir = args.out or default_out_dir(run_dirs, args.target)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_layers(samples, args)
    if not rows:
        raise SystemExit("No layers had enough labeled samples to fit a probe.")

    result_df = pd.DataFrame(rows).sort_values(["skipped", "test_accuracy", "train_accuracy"], ascending=[True, False, False])
    result_df.to_csv(out_dir / "linear_probe_results.csv", index=False)
    best_rows = result_df[result_df["skipped"] == False]  # noqa: E712
    best = best_rows.iloc[0].to_dict() if len(best_rows) else None
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "target": args.target,
                "feature": args.feature,
                "call_type": args.call_type,
                "t": args.t,
                "sample_count": len(samples),
                "run_dirs": [str(path) for path in run_dirs],
                "best_layer": best,
            },
            handle,
            indent=2,
            default=str,
        )
    pd.DataFrame(
        [
            {
                "run_dir": str(sample["run_dir"]),
                "call_id": sample["call_id"],
                "layer_count": len(sample["layers"]),
                "label": sample["label"],
                "agent_id": sample["metadata"].get("agent_id"),
                "t": sample["metadata"].get("t"),
                "truth_country": sample["metadata"].get("truth_country"),
                "predicted_country": sample["probe_row"].get("country") if sample["probe_row"] else None,
            }
            for sample in samples
        ]
    ).to_csv(out_dir / "samples.csv", index=False)
    print(f"Wrote linear-probe results to {out_dir}")
    if best is not None:
        print(
            "Best layer "
            f"{best['layer']} test_accuracy={best['test_accuracy']} "
            f"train_accuracy={best['train_accuracy']}"
        )


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


def default_out_dir(run_dirs: list[Path], target: str) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / f"linear_probe_{target}"
    common_parent = run_dirs[0].parent
    return common_parent / f"linear_probe_{target}"


def load_samples(run_dirs: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        index_rows = read_jsonl(run_dir / "activations" / "index.jsonl")
        probe_rows = {
            (int(row.get("t", -1)), int(row.get("agent_id", -1))): row
            for row in read_jsonl(run_dir / "probes.jsonl")
        }
        manifest = read_json(run_dir / "trial_manifest.json")
        crop_diagnostics = {
            int(row["agent_id"]): row
            for row in manifest.get("crop_diagnostics", [])
            if "agent_id" in row
        }
        for row in index_rows:
            if row.get("call_type") != args.call_type:
                continue
            if int(row.get("t", -1)) != args.t:
                continue
            agent_id = int(row.get("agent_id", -1))
            probe_row = probe_rows.get((int(row.get("t", -1)), agent_id))
            crop_diagnostic = row.get("crop_diagnostic") or crop_diagnostics.get(agent_id)
            label = label_for_target(args.target, row, probe_row, crop_diagnostic)
            if label is None:
                continue
            tensor_path = resolve_tensor_path(run_dir, row)
            payload = load_tensor_payload(tensor_path)
            if args.feature not in payload:
                raise KeyError(f"{tensor_path} does not contain feature {args.feature!r}")
            feature_tensor = payload[args.feature].detach().float().cpu().numpy()
            layers = [int(value) for value in payload["layers"].detach().cpu().numpy().tolist()]
            if args.layers is not None:
                wanted = set(args.layers)
                keep_indices = [idx for idx, layer in enumerate(layers) if layer in wanted]
                layers = [layers[idx] for idx in keep_indices]
                feature_tensor = feature_tensor[keep_indices, :]
            samples.append(
                {
                    "run_dir": run_dir,
                    "call_id": row.get("call_id"),
                    "metadata": row,
                    "probe_row": probe_row,
                    "crop_diagnostic": crop_diagnostic,
                    "label": str(label),
                    "layers": layers,
                    "features": feature_tensor,
                }
            )
    return samples


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


def label_for_target(
    target: str,
    activation_row: dict[str, Any],
    probe_row: dict[str, Any] | None,
    crop_diagnostic: dict[str, Any] | None,
) -> Any:
    if target == "truth_country":
        return activation_row.get("truth_country")
    if target == "predicted_country":
        return probe_row.get("country") if probe_row else None
    if target == "correct":
        return probe_row.get("correct") if probe_row else None
    if crop_diagnostic is None:
        return None
    if target == "truth_compatible":
        return crop_diagnostic.get("truth_compatible")
    if target == "is_unique":
        return crop_diagnostic.get("is_unique")
    if target == "informativeness_label":
        return crop_diagnostic.get("informativeness_label")
    if target == "compatible_country_count":
        return crop_diagnostic.get("compatible_country_count")
    raise ValueError(f"Unsupported target: {target}")


def evaluate_layers(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_layer: dict[int, list[tuple[np.ndarray, str]]] = defaultdict(list)
    for sample in samples:
        for idx, layer in enumerate(sample["layers"]):
            by_layer[layer].append((sample["features"][idx], sample["label"]))

    rows: list[dict[str, Any]] = []
    for layer, pairs in sorted(by_layer.items()):
        X = np.stack([feature for feature, _ in pairs]).astype(np.float32)
        y = [label for _, label in pairs]
        result = fit_ridge_probe(
            X,
            y,
            alpha=args.alpha,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        rows.append(
            {
                "layer": layer,
                "sample_count": len(y),
                "class_count": len(set(y)),
                **result,
            }
        )
    return rows


def fit_ridge_probe(
    X: np.ndarray,
    labels: list[str],
    *,
    alpha: float,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    classes = sorted(set(labels))
    if len(classes) < 2:
        return skipped_result("need at least two classes")
    train_idx, test_idx = stratified_split(labels, test_fraction=test_fraction, seed=seed)
    if len(train_idx) < len(classes):
        return skipped_result("not enough training samples for all classes")

    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_train = np.array([class_to_idx[labels[idx]] for idx in train_idx], dtype=np.int64)
    X_train = X[train_idx]
    X_test = X[test_idx] if test_idx else np.empty((0, X.shape[1]), dtype=np.float32)
    y_test = np.array([class_to_idx[labels[idx]] for idx in test_idx], dtype=np.int64)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std if len(test_idx) else X_test

    X_train_aug = np.concatenate([X_train_std, np.ones((X_train_std.shape[0], 1), dtype=np.float32)], axis=1)
    Y = np.zeros((X_train_aug.shape[0], len(classes)), dtype=np.float32)
    Y[np.arange(X_train_aug.shape[0]), y_train] = 1.0
    penalty = np.eye(X_train_aug.shape[1], dtype=np.float32) * float(alpha)
    penalty[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(X_train_aug.T @ X_train_aug + penalty, X_train_aug.T @ Y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(X_train_aug.T @ X_train_aug + penalty) @ X_train_aug.T @ Y

    train_pred = np.argmax(X_train_aug @ weights, axis=1)
    train_accuracy = float(np.mean(train_pred == y_train))
    if len(test_idx):
        X_test_aug = np.concatenate([X_test_std, np.ones((X_test_std.shape[0], 1), dtype=np.float32)], axis=1)
        test_pred = np.argmax(X_test_aug @ weights, axis=1)
        test_accuracy = float(np.mean(test_pred == y_test))
    else:
        test_accuracy = None

    return {
        "skipped": False,
        "skip_reason": None,
        "train_sample_count": len(train_idx),
        "test_sample_count": len(test_idx),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classes": json.dumps(classes, ensure_ascii=True),
    }


def skipped_result(reason: str) -> dict[str, Any]:
    return {
        "skipped": True,
        "skip_reason": reason,
        "train_sample_count": 0,
        "test_sample_count": 0,
        "train_accuracy": None,
        "test_accuracy": None,
        "classes": "[]",
    }


def stratified_split(
    labels: list[str],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for indices in by_label.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) >= 2:
            test_count = max(1, int(round(len(shuffled) * test_fraction)))
            test_count = min(test_count, len(shuffled) - 1)
        else:
            test_count = 0
        test_idx.extend(shuffled[:test_count])
        train_idx.extend(shuffled[test_count:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


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
