#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize open-model scale comparison runs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out or (root / "scale_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs(root)
    if not model_dirs:
        raise SystemExit(f"No model run directories with batch_summary.csv under {root}")

    batch_rows: list[pd.DataFrame] = []
    agent_rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        batch_path = model_dir / "batch_summary.csv"
        batch_df = pd.read_csv(batch_path)
        model_name = infer_model_name(model_dir, batch_df)
        batch_df.insert(0, "model_run_dir", str(model_dir))
        batch_df.insert(1, "model_name", model_name)
        batch_df.insert(2, "model_scale_b", infer_scale_b(model_name))
        batch_rows.append(batch_df)
        agent_rows.extend(load_initial_probe_agent_rows(model_dir, model_name))

    combined_batch = pd.concat(batch_rows, ignore_index=True)
    combined_batch.to_csv(out_dir / "model_seed_summary.csv", index=False)

    agent_df = pd.DataFrame(agent_rows)
    if not agent_df.empty:
        agent_df.to_csv(out_dir / "initial_probe_agent_rows.csv", index=False)
        probe_metrics = summarize_initial_probe_agents(agent_df)
        probe_metrics.to_csv(out_dir / "initial_probe_metrics.csv", index=False)
        country_dist = country_distribution(agent_df)
        country_dist.to_csv(out_dir / "initial_probe_country_distribution.csv", index=False)
    else:
        probe_metrics = pd.DataFrame()
        country_dist = pd.DataFrame()

    model_metrics = summarize_batch(combined_batch, probe_metrics)
    model_metrics.to_csv(out_dir / "model_scale_metrics.csv", index=False)

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "root": str(root),
                "model_dirs": [str(path) for path in model_dirs],
                "outputs": [
                    "model_scale_metrics.csv",
                    "model_seed_summary.csv",
                    "initial_probe_metrics.csv",
                    "initial_probe_country_distribution.csv",
                    "initial_probe_agent_rows.csv",
                ],
            },
            handle,
            indent=2,
        )
    print(f"Wrote model-scale summary to {out_dir}")
    print((out_dir / "model_scale_metrics.csv").read_text())


def discover_model_dirs(root: Path) -> list[Path]:
    candidates = []
    if (root / "batch_summary.csv").exists():
        candidates.append(root)
    candidates.extend(path.parent for path in root.glob("*/batch_summary.csv"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in sorted(candidates):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def infer_model_name(model_dir: Path, batch_df: pd.DataFrame) -> str:
    if "model" in batch_df and batch_df["model"].notna().any():
        return str(batch_df["model"].dropna().iloc[0])
    return model_dir.name


def infer_scale_b(model_name: str) -> float | None:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    return float(match.group(1)) if match else None


def load_initial_probe_agent_rows(model_dir: Path, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(model_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        probes = read_jsonl(seed_dir / "probes.jsonl")
        manifest = read_json(seed_dir / "trial_manifest.json")
        crop_diagnostics = {
            int(row["agent_id"]): row
            for row in manifest.get("crop_diagnostics", [])
            if "agent_id" in row
        }
        truth_country = manifest.get("truth_country")
        for probe in probes:
            if int(probe.get("t", -1)) != 0:
                continue
            agent_id = int(probe.get("agent_id", -1))
            diagnostic = crop_diagnostics.get(agent_id, {})
            compatible = diagnostic.get("compatible_countries") or []
            if isinstance(compatible, str):
                try:
                    compatible = json.loads(compatible)
                except json.JSONDecodeError:
                    compatible = []
            country = probe.get("country")
            rows.append(
                {
                    "model_name": model_name,
                    "model_scale_b": infer_scale_b(model_name),
                    "model_run_dir": str(model_dir),
                    "seed_dir": seed_dir.name,
                    "seed": parse_seed(seed_dir.name),
                    "agent_id": agent_id,
                    "truth_country": truth_country,
                    "predicted_country": country,
                    "valid": bool(probe.get("valid", False)),
                    "correct": bool(probe.get("correct", False)),
                    "is_france": country == "France",
                    "truth_compatible": diagnostic.get("truth_compatible"),
                    "france_compatible": "France" in compatible,
                    "predicted_country_compatible": country in compatible if country else None,
                    "compatible_country_count": diagnostic.get("compatible_country_count"),
                    "informativeness_label": diagnostic.get("informativeness_label"),
                    "is_unique": diagnostic.get("is_unique"),
                }
            )
    return rows


def parse_seed(name: str) -> int | None:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return None


def summarize_initial_probe_agents(agent_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in agent_df.groupby("model_name", dropna=False):
        valid = group[group["valid"] == True]  # noqa: E712
        france_incompatible = valid[valid["france_compatible"] == False]  # noqa: E712
        rows.append(
            {
                "model_name": model_name,
                "model_scale_b": group["model_scale_b"].dropna().iloc[0]
                if group["model_scale_b"].notna().any()
                else None,
                "initial_probe_count": len(group),
                "initial_valid_probe_count": len(valid),
                "initial_agent_accuracy": float(valid["correct"].mean()) if len(valid) else None,
                "initial_agent_france_rate": float(valid["is_france"].mean()) if len(valid) else None,
                "initial_agent_predicted_compatible_rate": float(valid["predicted_country_compatible"].mean())
                if len(valid)
                else None,
                "france_when_france_incompatible_rate": float(france_incompatible["is_france"].mean())
                if len(france_incompatible)
                else None,
                "mean_compatible_country_count": float(pd.to_numeric(valid["compatible_country_count"]).mean())
                if len(valid)
                else None,
            }
        )
    return pd.DataFrame(rows)


def country_distribution(agent_df: pd.DataFrame) -> pd.DataFrame:
    valid = agent_df[agent_df["valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return pd.DataFrame()
    counts = (
        valid.groupby(["model_name", "predicted_country"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("model_name")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts.sort_values(["model_name", "count"], ascending=[True, False])


def summarize_batch(batch_df: pd.DataFrame, probe_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in batch_df.groupby("model_name", dropna=False):
        final_vote = group["final_vote_country"].fillna("")
        initial_vote = group["initial_vote_country"].fillna("")
        rows.append(
            {
                "model_name": model_name,
                "model_scale_b": group["model_scale_b"].dropna().iloc[0]
                if group["model_scale_b"].notna().any()
                else None,
                "seed_count": len(group),
                "initial_accuracy_mean": float(group["initial_accuracy"].mean()),
                "final_accuracy_mean": float(group["final_accuracy"].mean()),
                "oracle_accuracy_mean": float(group["oracle_accuracy"].mean())
                if "oracle_accuracy" in group
                else None,
                "initial_vote_france_rate": float((initial_vote == "France").mean()),
                "final_vote_france_rate": float((final_vote == "France").mean()),
                "correct_consensus_rate": float(group["final_consensus_correct"].fillna(False).mean()),
                "wrong_consensus_rate": float((group["final_outcome"] == "wrong_consensus").mean()),
                "unique_final_vote_countries": int(final_vote[final_vote != ""].nunique()),
                "mean_initial_compatible_country_count": float(
                    group["initial_mean_compatible_country_count"].mean()
                )
                if "initial_mean_compatible_country_count" in group
                else None,
            }
        )
    metrics = pd.DataFrame(rows)
    if not probe_metrics.empty:
        metrics = metrics.merge(
            probe_metrics,
            on=["model_name", "model_scale_b"],
            how="left",
        )
    return metrics.sort_values(["model_scale_b", "model_name"], na_position="last")


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
