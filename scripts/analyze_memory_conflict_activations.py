#!/usr/bin/env python3
"""Analyze layer-wise concept geometry for memory-conflict probe activations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


DEFAULT_CONCEPTS = [
    "private_evidence_strength",
    "social_evidence_type",
    "crop_condition",
    "lure_relation",
    "false_memory_bin",
    "memory_majority",
    "response_type",
    "alignment_type",
    "choice_axis",
    "chose_lure",
    "chose_truth",
    "correct",
    "truth_country",
    "lure_country",
    "choice_country",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize concept clustering/separation across layers for the "
            "single-agent private-vs-social memory-conflict probe."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directory, batch directory, or directories containing activations/index.jsonl.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--feature",
        choices=("last_prompt_token", "mean_prompt"),
        default="last_prompt_token",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer ids to include. Defaults to every captured layer.",
    )
    parser.add_argument(
        "--concept",
        action="append",
        default=None,
        help="Concept column to analyze. Can be repeated. Defaults to the phase-3 concept set.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=2,
        help="Minimum examples per class for centroid geometry.",
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

    concepts = args.concept or list(DEFAULT_CONCEPTS)
    out_dir = args.out or default_out_dir(run_dirs, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = sample_table(samples)
    concept_rows, centroid_rows = analyze_concepts(
        samples,
        concepts=concepts,
        min_class_count=args.min_class_count,
    )

    pd.DataFrame(sample_rows).to_csv(out_dir / "activation_samples.csv", index=False)
    pd.DataFrame(concept_rows).to_csv(out_dir / "concept_separation_by_layer.csv", index=False)
    pd.DataFrame(centroid_rows).to_csv(out_dir / "centroid_similarity_by_layer.csv", index=False)
    write_notes(out_dir / "analysis_notes.md")
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "feature": args.feature,
                "run_dirs": [str(path) for path in run_dirs],
                "sample_count": len(samples),
                "concepts": concepts,
                "min_class_count": args.min_class_count,
                "outputs": [
                    "activation_samples.csv",
                    "concept_separation_by_layer.csv",
                    "centroid_similarity_by_layer.csv",
                    "analysis_notes.md",
                ],
            },
            handle,
            indent=2,
        )
    print(f"Wrote memory-conflict activation analysis to {out_dir}")
    print(f"Samples: {len(samples)} | concept/layer rows: {len(concept_rows)}")


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
    name = f"memory_conflict_activation_concepts_{args.feature}"
    if len(run_dirs) == 1:
        return run_dirs[0] / name
    return run_dirs[0].parent / name


def load_samples(run_dirs: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    wanted_layers = set(args.layers) if args.layers is not None else None
    for run_dir in run_dirs:
        result_rows = load_result_rows(run_dir)
        for index_row in read_jsonl(run_dir / "activations" / "index.jsonl"):
            if str(index_row.get("call_type", "")) != "probe":
                continue
            payload = load_tensor_payload(resolve_tensor_path(run_dir, index_row))
            if args.feature not in payload:
                raise KeyError(f"{index_row.get('tensor_path')} does not contain {args.feature!r}")
            layers = tensor_to_list(payload["layers"])
            features = payload[args.feature].detach().float().cpu().numpy()
            if wanted_layers is not None:
                keep = [idx for idx, layer in enumerate(layers) if layer in wanted_layers]
                layers = [layers[idx] for idx in keep]
                features = features[keep, :]
            if not layers:
                continue
            trial_id = str(index_row.get("trial_id", ""))
            labels = dict(index_row)
            if trial_id in result_rows:
                labels.update(result_rows[trial_id])
            labels.update(derived_labels(labels))
            samples.append(
                {
                    "run_dir": run_dir,
                    "run_id": run_dir.name,
                    "call_id": index_row.get("call_id"),
                    "trial_id": trial_id,
                    "model": labels.get("model"),
                    "layers": layers,
                    "features": features,
                    "labels": labels,
                }
            )
    return samples


def load_result_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    csv_path = run_dir / "results.csv"
    jsonl_path = run_dir / "results.jsonl"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    else:
        return {}
    if df.empty or "trial_id" not in df:
        return {}
    df = ensure_response_type_columns(df)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rows[str(row["trial_id"])] = {key: normalize_scalar(value) for key, value in row.to_dict().items()}
    return rows


def ensure_response_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "response_type" not in df:
        response_types = []
        alignment_types = []
        for _, row in df.iterrows():
            if not bool_value(row.get("valid")):
                response_types.append(None)
                alignment_types.append(None)
            elif bool_value(row.get("chose_truth")):
                response_types.append("private_evidence")
                alignment_types.append("private_evidence")
            elif bool_value(row.get("chose_lure")):
                response_types.append("social_evidence")
                alignment_types.append("social_evidence")
            elif bool_value(row.get("chose_other")):
                response_types.append("other")
                alignment_types.append("other")
            else:
                response_types.append("unknown")
                alignment_types.append("unknown")
        df["response_type"] = response_types
        df["alignment_type"] = alignment_types
    return df


def derived_labels(labels: dict[str, Any]) -> dict[str, Any]:
    false_count = int_or_none(labels.get("false_memory_count"))
    true_count = int_or_none(labels.get("true_memory_count"))
    memory_total = int_or_none(labels.get("memory_total"))
    if memory_total is None and false_count is not None and true_count is not None:
        memory_total = false_count + true_count
    false_memory_bin = None
    memory_majority = None
    if false_count is not None and true_count is not None:
        if false_count == 0:
            false_memory_bin = "all_private_memory"
        elif memory_total is not None and false_count == memory_total:
            false_memory_bin = "all_social_memory"
        elif false_count < true_count:
            false_memory_bin = "private_memory_majority"
        elif false_count > true_count:
            false_memory_bin = "social_memory_majority"
        else:
            false_memory_bin = "memory_tie"
        if false_count > true_count:
            memory_majority = "social_evidence"
        elif true_count > false_count:
            memory_majority = "private_evidence"
        else:
            memory_majority = "tie"
    choice_axis = None
    if bool_value(labels.get("chose_truth")):
        choice_axis = "private_target"
    elif bool_value(labels.get("chose_lure")):
        choice_axis = "social_lure"
    elif bool_value(labels.get("valid")):
        choice_axis = "other"
    return {
        "false_memory_bin": false_memory_bin,
        "memory_majority": memory_majority,
        "choice_axis": choice_axis,
    }


def sample_table(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        labels = sample["labels"]
        rows.append(
            {
                "run_dir": str(sample["run_dir"]),
                "run_id": sample["run_id"],
                "call_id": sample["call_id"],
                "trial_id": sample["trial_id"],
                "model": sample["model"],
                "layer_count": len(sample["layers"]),
                "layers": json.dumps(sample["layers"]),
                "m": labels.get("m"),
                "truth_country": labels.get("truth_country"),
                "lure_country": labels.get("lure_country"),
                "choice_country": labels.get("choice_country"),
                "private_evidence_strength": labels.get("private_evidence_strength"),
                "social_evidence_type": labels.get("social_evidence_type"),
                "false_memory_count": labels.get("false_memory_count"),
                "true_memory_count": labels.get("true_memory_count"),
                "false_memory_bin": labels.get("false_memory_bin"),
                "memory_majority": labels.get("memory_majority"),
                "response_type": labels.get("response_type"),
                "alignment_type": labels.get("alignment_type"),
                "choice_axis": labels.get("choice_axis"),
                "correct": labels.get("correct"),
            }
        )
    return rows


def analyze_concepts(
    samples: list[dict[str, Any]],
    *,
    concepts: list[str],
    min_class_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_values = sorted({layer for sample in samples for layer in sample["layers"]})
    concept_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    for concept in concepts:
        for layer in layer_values:
            layer_examples = examples_for_layer(samples, layer=layer, concept=concept)
            counts = Counter(label for label, _vec in layer_examples)
            labels = sorted(label for label, count in counts.items() if count >= min_class_count)
            if len(labels) < 2:
                continue
            filtered = [(label, vec) for label, vec in layer_examples if label in labels]
            centroids = {
                label: np.mean([vec for item_label, vec in filtered if item_label == label], axis=0)
                for label in labels
            }
            within_values = [
                cosine(vec, centroids[label])
                for label, vec in filtered
                if label in centroids
            ]
            pairwise_rows = []
            between_values = []
            for left_index, left_label in enumerate(labels):
                for right_label in labels[left_index + 1 :]:
                    left_centroid = centroids[left_label]
                    right_centroid = centroids[right_label]
                    similarity = cosine(left_centroid, right_centroid)
                    distance = l2_distance(left_centroid, right_centroid)
                    between_values.append(similarity)
                    pairwise_rows.append(
                        {
                            "concept": concept,
                            "layer": layer,
                            "label_a": left_label,
                            "label_b": right_label,
                            "count_a": counts[left_label],
                            "count_b": counts[right_label],
                            "centroid_cosine_similarity": similarity,
                            "centroid_l2_distance": distance,
                        }
                    )
            nearest_accuracy = nearest_centroid_accuracy(filtered, centroids)
            concept_rows.append(
                {
                    "concept": concept,
                    "layer": layer,
                    "sample_count": len(filtered),
                    "class_count": len(labels),
                    "classes": json.dumps(labels),
                    "class_counts": json.dumps({label: counts[label] for label in labels}, sort_keys=True),
                    "mean_within_class_cosine_to_centroid": mean_or_none(within_values),
                    "mean_between_centroid_cosine": mean_or_none(between_values),
                    "concept_separation_cosine_gap": diff_or_none(
                        mean_or_none(within_values),
                        mean_or_none(between_values),
                    ),
                    "nearest_centroid_accuracy": nearest_accuracy,
                    "mean_pairwise_centroid_l2_distance": mean_or_none(
                        [row["centroid_l2_distance"] for row in pairwise_rows]
                    ),
                }
            )
            centroid_rows.extend(pairwise_rows)
    return concept_rows, centroid_rows


def examples_for_layer(
    samples: list[dict[str, Any]],
    *,
    layer: int,
    concept: str,
) -> list[tuple[str, np.ndarray]]:
    examples: list[tuple[str, np.ndarray]] = []
    for sample in samples:
        if layer not in sample["layers"]:
            continue
        label = label_value(sample["labels"].get(concept))
        if label is None:
            continue
        layer_index = sample["layers"].index(layer)
        examples.append((label, sample["features"][layer_index]))
    return examples


def nearest_centroid_accuracy(
    examples: list[tuple[str, np.ndarray]],
    centroids: dict[str, np.ndarray],
) -> float | None:
    if not examples:
        return None
    correct = 0
    for label, vec in examples:
        prediction = max(
            centroids,
            key=lambda candidate: cosine(vec, centroids[candidate]),
        )
        correct += int(prediction == label)
    return float(correct / len(examples))


def label_value(value: Any) -> str | None:
    value = normalize_scalar(value)
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        return text
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def bool_value(value: Any) -> bool:
    value = normalize_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def int_or_none(value: Any) -> int | None:
    value = normalize_scalar(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def tensor_to_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        return [int(item) for item in value.detach().cpu().numpy().tolist()]
    return [int(item) for item in value]


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


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def mean_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and not np.isnan(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def diff_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


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


def write_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Memory-Conflict Activation Concept Notes",
                "",
                "The main score is `concept_separation_cosine_gap`: mean cosine to a sample's own class centroid minus mean cosine between class centroids.",
                "",
                "Interpretation:",
                "",
                "- Larger positive gaps mean a concept is more geometrically clustered at that layer.",
                "- `private_evidence_strength` and `social_evidence_type` test stimulus-side concepts.",
                "- `false_memory_bin` and `memory_majority` test social-evidence composition.",
                "- `response_type` and `choice_axis` test decision-side concepts.",
                "- Compare early, middle, and late layers before making claims about broad-to-structured representation changes.",
                "",
                "Nearest-centroid accuracy is descriptive, not a trained probe. It is included only as a simple geometry readout.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
