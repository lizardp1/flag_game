#!/usr/bin/env python3
"""Draw the final Mechanism 1 random-field figure with an empirical field bridge.

This script builds the four-panel figure:

  A. empirical crop field h_i
  B. abstract private-field mixture by N
  C. abstract finite-agent endpoint regimes
  D. N x J mean-field phase diagram

It reuses the repeated isolated crop-probe table produced by
scripts/run_flag_empirical_crop_field_probes.py for Panel A only. Positive h is
truth-positive:

    h_i = log((p_i(T) + eps) / (p_i(R) + eps)).

Panels B--D use a country-free random-field QSG simulation. They do not use
country identities, flag images, or LLM outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
import sys
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))

import make_flag_game_empirical_crop_field_mechanism as base  # noqa: E402


DATA_DIR = ROOT / "paper" / "exports" / "data"
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DEFAULT_RUN_ROOT = base.DEFAULT_MECHANISM_RUN_ROOT
DEFAULT_N_VALUES = base.DEFAULT_N_VALUES
ENDPOINT_ORDER = ["correct_consensus", "wrong_consensus", "polarization", "fragmentation"]
ENDPOINT_LABELS = {
    "correct_consensus": "Correct consensus",
    "wrong_consensus": "Wrong consensus",
    "polarization": "Polarization",
    "fragmentation": "Fragmentation",
}
ENDPOINT_COLORS = {
    "correct_consensus": base.GREEN,
    "wrong_consensus": base.ORANGE,
    "polarization": base.PURPLE,
    "fragmentation": base.LIGHT_GRAY,
}
FIELD_TYPE_ORDER = ["T", "R", "0"]
FIELD_TYPE_LABELS = {
    "T": r"truth-supporting $h_i>\theta$",
    "R": r"rival-supporting $h_i<-\theta$",
    "0": r"ambiguous $|h_i|\leq\theta$",
}
FIELD_TYPE_COLORS = {
    "T": base.GREEN,
    "R": base.ORANGE,
    "0": base.GRAY,
}


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    return values


def default_j_grid() -> list[float]:
    return [round(0.10 * value, 2) for value in range(0, 21)]


def seed_sort_key(path: Path) -> tuple[int, str]:
    parsed = base.seed_from_path(path)
    return (parsed if parsed is not None else 10**9, path.name)


def load_truth(seed_dir: Path, manifest: dict[str, Any]) -> str:
    summary_path = seed_dir / "summary.json"
    summary = base.read_json(summary_path) if summary_path.exists() else {}
    return base.normalize_text(
        manifest.get("truth_country")
        or manifest.get("target_country")
        or summary.get("truth_country")
        or summary.get("target_country")
    )


def field_lookup(fields: list[dict[str, Any]]) -> dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]]:
    lookup: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]] = {}
    for row in fields:
        key = base.rounded_geometry_key(row)
        if key is None:
            continue
        lookup[(base.country_key(row["target"]), key)] = row
    return lookup


def crop_set_from_assignments(
    *,
    assignments: list[dict[str, Any]],
    fields_by_target_geometry: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]],
    truth: str,
    n_value: int,
    condition: str,
    seed_name: str,
    allow_missing: bool,
) -> dict[str, Any] | None:
    h_values: list[float] = []
    p_truth_values: list[float] = []
    p_rival_values: list[float] = []
    missing: list[tuple[int, int, int, int]] = []
    matched_fields: list[dict[str, Any]] = []
    for assignment in assignments[:n_value]:
        geom = base.rounded_geometry_key(assignment)
        if geom is None:
            continue
        field = fields_by_target_geometry.get((base.country_key(truth), geom))
        if field is None:
            missing.append(geom)
            continue
        h_values.append(float(field["h"]))
        p_truth_values.append(float(field["p_truth"]))
        p_rival_values.append(float(field["p_rival"]))
        matched_fields.append(field)

    if missing and not allow_missing:
        preview = ", ".join(str(item) for item in missing[:6])
        raise SystemExit(
            f"Missing {len(missing)} measured crop fields for {truth} {seed_name} N={n_value}; "
            f"first missing {preview}. Rerun isolated crop probes for these manifests."
        )
    if len(h_values) != n_value:
        return None

    h_array = np.array(h_values, dtype=float)
    s0 = np.array(p_truth_values, dtype=float) - np.array(p_rival_values, dtype=float)
    return {
        "N": n_value,
        "condition": condition,
        "seed": seed_name,
        "target": truth,
        "rival": matched_fields[0]["rival"] if matched_fields else "",
        "n_agents": n_value,
        "h_values": h_values,
        "p_truth_values": p_truth_values,
        "p_rival_values": p_rival_values,
        "m0": float(np.mean(s0)),
        "q0": float(np.mean(s0 * s0)),
        "mean_h": float(np.mean(h_array)),
        "sd_h": float(np.std(h_array)),
    }


def load_exact_crop_sets(
    fields: list[dict[str, Any]],
    *,
    run_root: Path,
    condition: str,
    n_values: list[int],
    target_filter: str | None,
    seed_filter: set[str] | None,
    allow_missing: bool,
    max_seeds_per_n: int | None,
) -> list[dict[str, Any]]:
    fields_by_target_geometry = field_lookup(fields)
    crop_sets: list[dict[str, Any]] = []
    for n_value in n_values:
        condition_dir = run_root / f"N{n_value}" / condition
        seed_dirs = sorted(condition_dir.glob("seed_*"), key=seed_sort_key)
        if seed_filter is not None:
            seed_dirs = [path for path in seed_dirs if path.name in seed_filter]
        if max_seeds_per_n is not None:
            seed_dirs = seed_dirs[:max_seeds_per_n]
        for seed_dir in seed_dirs:
            manifest_path = seed_dir / "trial_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = base.read_json(manifest_path)
            truth = load_truth(seed_dir, manifest)
            if target_filter and base.country_key(truth) != base.country_key(target_filter):
                continue
            crop_set = crop_set_from_assignments(
                assignments=list(manifest.get("assignments") or []),
                fields_by_target_geometry=fields_by_target_geometry,
                truth=truth,
                n_value=n_value,
                condition=condition,
                seed_name=seed_dir.name,
                allow_missing=allow_missing,
            )
            if crop_set is not None:
                crop_sets.append(crop_set)
    return sorted(crop_sets, key=lambda row: (int(row["N"]), str(row["seed"])))


def load_nested_prefix_crop_sets(
    fields: list[dict[str, Any]],
    *,
    run_root: Path,
    condition: str,
    n_values: list[int],
    target_filter: str | None,
    seed_filter: set[str] | None,
    allow_missing: bool,
    max_seeds_per_n: int | None,
) -> list[dict[str, Any]]:
    fields_by_target_geometry = field_lookup(fields)
    max_n = max(n_values)
    condition_dir = run_root / f"N{max_n}" / condition
    seed_dirs = sorted(condition_dir.glob("seed_*"), key=seed_sort_key)
    if seed_filter is not None:
        seed_dirs = [path for path in seed_dirs if path.name in seed_filter]
    if max_seeds_per_n is not None:
        seed_dirs = seed_dirs[:max_seeds_per_n]

    crop_sets: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        manifest_path = seed_dir / "trial_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = base.read_json(manifest_path)
        truth = load_truth(seed_dir, manifest)
        if target_filter and base.country_key(truth) != base.country_key(target_filter):
            continue
        assignments = list(manifest.get("assignments") or [])
        for n_value in n_values:
            crop_set = crop_set_from_assignments(
                assignments=assignments,
                fields_by_target_geometry=fields_by_target_geometry,
                truth=truth,
                n_value=n_value,
                condition=condition,
                seed_name=seed_dir.name,
                allow_missing=allow_missing,
            )
            if crop_set is not None:
                crop_sets.append(crop_set)
    return sorted(crop_sets, key=lambda row: (int(row["N"]), str(row["seed"])))


def load_crop_sets(
    fields: list[dict[str, Any]],
    *,
    run_root: Path,
    condition: str,
    n_values: list[int],
    target_filter: str | None,
    seed_filter: set[str] | None,
    allow_missing: bool,
    max_seeds_per_n: int | None,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "nested_prefix":
        return load_nested_prefix_crop_sets(
            fields,
            run_root=run_root,
            condition=condition,
            n_values=n_values,
            target_filter=target_filter,
            seed_filter=seed_filter,
            allow_missing=allow_missing,
            max_seeds_per_n=max_seeds_per_n,
        )
    return load_exact_crop_sets(
        fields,
        run_root=run_root,
        condition=condition,
        n_values=n_values,
        target_filter=target_filter,
        seed_filter=seed_filter,
        allow_missing=allow_missing,
        max_seeds_per_n=max_seeds_per_n,
    )


def bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if values.size <= 1 or draws <= 0:
        value = float(values[0]) if values.size == 1 else float(np.mean(values))
        return value, value
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def field_balance_summary(
    crop_sets: list[dict[str, Any]],
    *,
    n_values: list[int],
    bootstrap_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for n_value in n_values:
        sets = [row for row in crop_sets if int(row["N"]) == n_value]
        if not sets:
            continue
        m_values = np.array([float(row["m0"]) for row in sets], dtype=float)
        q_values = np.array([float(row["q0"]) for row in sets], dtype=float)
        m_low, m_high = bootstrap_ci(m_values, rng=rng, draws=bootstrap_draws)
        q_low, q_high = bootstrap_ci(q_values, rng=rng, draws=bootstrap_draws)
        rows.append(
            {
                "N": n_value,
                "n_seed_sets": len(sets),
                "m0_mean": float(np.mean(m_values)),
                "m0_low": m_low,
                "m0_high": m_high,
                "q0_mean": float(np.mean(q_values)),
                "q0_low": q_low,
                "q0_high": q_high,
            }
        )
    return rows


def endpoint_from_truth_share(truth_share: np.ndarray) -> np.ndarray:
    rival_share = 1.0 - truth_share
    out = np.full(truth_share.shape, "fragmentation", dtype=object)
    out[truth_share >= 0.85] = "correct_consensus"
    out[rival_share >= 0.85] = "wrong_consensus"
    polarized = (
        (truth_share < 0.85)
        & (rival_share < 0.85)
        & (truth_share >= 0.25)
        & (rival_share >= 0.25)
    )
    out[polarized] = "polarization"
    return out


def endpoint_from_vote_shares(truth_share: np.ndarray, rival_share: np.ndarray) -> np.ndarray:
    out = np.full(truth_share.shape, "fragmentation", dtype=object)
    out[truth_share >= 0.85] = "correct_consensus"
    out[rival_share >= 0.85] = "wrong_consensus"
    polarized = (
        (truth_share < 0.85)
        & (rival_share < 0.85)
        & (truth_share >= 0.25)
        & (rival_share >= 0.25)
    )
    out[polarized] = "polarization"
    return out


def normal_cdf(value: float, *, mean: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if value >= mean else 0.0
    z = (value - mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def cutoff_field_masses(
    masses: dict[str, Any],
    *,
    theta: float,
    h_truth: float,
    h_rival: float,
    h_ambiguous: float,
    sigma_truth: float,
    sigma_rival: float,
    sigma_ambiguous: float,
) -> dict[str, float]:
    weights = [float(masses["w_T"]), float(masses["w_R"]), float(masses["w_0"])]
    means = [h_truth, h_rival, h_ambiguous]
    sigmas = [sigma_truth, sigma_rival, sigma_ambiguous]
    truth_mass = 0.0
    rival_mass = 0.0
    ambiguous_mass = 0.0
    for weight, mean, sigma in zip(weights, means, sigmas):
        lower = normal_cdf(-theta, mean=mean, sigma=sigma)
        upper = normal_cdf(theta, mean=mean, sigma=sigma)
        rival_mass += weight * lower
        ambiguous_mass += weight * max(0.0, upper - lower)
        truth_mass += weight * max(0.0, 1.0 - upper)
    total = truth_mass + rival_mass + ambiguous_mass
    if total > 0:
        truth_mass /= total
        rival_mass /= total
        ambiguous_mass /= total
    return {
        "cutoff_w_T": float(truth_mass),
        "cutoff_w_R": float(rival_mass),
        "cutoff_w_0": float(ambiguous_mass),
    }


def abstract_field_masses(n_value: float, *, max_informative_mass: float) -> dict[str, Any]:
    n_float = float(n_value)
    raw_truth = 0.68 * (1.0 - math.exp(-n_float / 8.0)) * math.exp(-n_float / 500.0)
    raw_rival = 0.56 * (1.0 - math.exp(-((n_float / 45.0) ** 1.7)))
    truth = raw_truth
    rival = raw_rival
    rescale = 1.0
    if truth + rival > max_informative_mass:
        rescale = max_informative_mass / (truth + rival)
        truth *= rescale
        rival *= rescale
    ambiguous = max(0.0, 1.0 - truth - rival)
    return {
        "N": int(n_float) if abs(n_float - round(n_float)) < 1e-9 else n_float,
        "w_T": float(truth),
        "w_R": float(rival),
        "w_0": float(ambiguous),
        "raw_w_T": float(raw_truth),
        "raw_w_R": float(raw_rival),
        "rescale": float(rescale),
    }


def empirical_anchor_masses(fields: list[dict[str, Any]], *, theta: float) -> dict[str, float]:
    h_values = np.array([float(row["h"]) for row in fields], dtype=float)
    if h_values.size == 0:
        raise ValueError("Need at least one empirical field for empirical-anchor mixture.")
    truth = float(np.mean(h_values > theta))
    rival = float(np.mean(h_values < -theta))
    ambiguous = max(0.0, 1.0 - truth - rival)
    return {"w_T": truth, "w_R": rival, "w_0": ambiguous}


def override_anchor_masses(
    anchor_masses: dict[str, float],
    *,
    truth_mass: float | None,
    rival_mass: float | None,
    ambiguous_mass: float | None,
) -> dict[str, float]:
    truth = float(anchor_masses["w_T"] if truth_mass is None else truth_mass)
    rival = float(anchor_masses["w_R"] if rival_mass is None else rival_mass)
    if ambiguous_mass is None:
        ambiguous = max(0.0, 1.0 - truth - rival)
    else:
        ambiguous = float(ambiguous_mass)
    total = truth + rival + ambiguous
    if total <= 0:
        raise ValueError("Anchor masses must have positive total mass.")
    if total > 1.0 + 1e-9:
        truth /= total
        rival /= total
        ambiguous /= total
    else:
        ambiguous = 1.0 - truth - rival
    return {"w_T": float(truth), "w_R": float(rival), "w_0": float(ambiguous)}


def empirical_anchor_field_masses(
    n_value: float,
    *,
    anchor_masses: dict[str, float],
    truth_tau: float,
    rival_tau: float,
    rival_power: float,
) -> dict[str, Any]:
    n_float = float(n_value)
    truth = float(anchor_masses["w_T"]) * (1.0 - math.exp(-n_float / truth_tau))
    rival = float(anchor_masses["w_R"]) * (1.0 - math.exp(-((n_float / rival_tau) ** rival_power)))
    ambiguous = max(0.0, 1.0 - truth - rival)
    return {
        "N": int(n_float) if abs(n_float - round(n_float)) < 1e-9 else n_float,
        "w_T": float(truth),
        "w_R": float(rival),
        "w_0": float(ambiguous),
        "raw_w_T": float(truth),
        "raw_w_R": float(rival),
        "rescale": 1.0,
    }


def abstract_mixture_rows(
    n_values: list[int],
    *,
    mode: str,
    max_informative_mass: float,
    anchor_masses: dict[str, float] | None = None,
    empirical_truth_tau: float,
    empirical_rival_tau: float,
    empirical_rival_power: float,
) -> list[dict[str, Any]]:
    if mode == "stylized":
        return [abstract_field_masses(n_value, max_informative_mass=max_informative_mass) for n_value in n_values]
    if mode == "empirical_anchor":
        if anchor_masses is None:
            raise ValueError("empirical_anchor mixture mode requires anchor_masses.")
        return [
            empirical_anchor_field_masses(
                n_value,
                anchor_masses=anchor_masses,
                truth_tau=empirical_truth_tau,
                rival_tau=empirical_rival_tau,
                rival_power=empirical_rival_power,
            )
            for n_value in n_values
        ]
    raise ValueError(f"Unknown mixture mode: {mode}")


def mean_field_fixed_point(
    *,
    masses: dict[str, Any],
    j_value: float,
    beta: float,
    h_truth: float,
    h_rival: float,
    h_ambiguous: float,
    initial_m: float | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 10000,
) -> tuple[float, float, int]:
    weights = np.array([masses["w_T"], masses["w_R"], masses["w_0"]], dtype=float)
    fields = np.array([h_truth, h_rival, h_ambiguous], dtype=float)
    if initial_m is None:
        initial_responses = np.tanh(beta * fields)
        m_value = float(np.sum(weights * initial_responses))
    else:
        m_value = float(initial_m)
    for iteration in range(max_iter):
        responses = np.tanh(beta * (fields + j_value * m_value))
        m_next = float(np.sum(weights * responses))
        if abs(m_next - m_value) < tolerance:
            m_value = m_next
            break
        m_value = m_next
    responses = np.tanh(beta * (fields + j_value * m_value))
    q_value = float(np.sum(weights * responses * responses))
    return m_value, q_value, iteration + 1


def classify_mean_field_phase(m_value: float, q_value: float, *, m_threshold: float, q_threshold: float) -> str:
    if q_value <= q_threshold:
        return "fragmentation"
    if m_value > m_threshold:
        return "correct_consensus"
    if m_value < -m_threshold:
        return "wrong_consensus"
    return "polarization"


def mean_field_phase_rows(
    n_grid: np.ndarray,
    *,
    mixture_mode: str,
    j_values: list[float],
    max_informative_mass: float,
    anchor_masses: dict[str, float] | None,
    empirical_truth_tau: float,
    empirical_rival_tau: float,
    empirical_rival_power: float,
    beta: float,
    h_truth: float,
    h_rival: float,
    h_ambiguous: float,
    initial_m: float | None,
    m_threshold: float,
    q_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_value in n_grid:
        if mixture_mode == "stylized":
            masses = abstract_field_masses(float(n_value), max_informative_mass=max_informative_mass)
        else:
            if anchor_masses is None:
                raise ValueError("empirical_anchor mixture mode requires anchor_masses.")
            masses = empirical_anchor_field_masses(
                float(n_value),
                anchor_masses=anchor_masses,
                truth_tau=empirical_truth_tau,
                rival_tau=empirical_rival_tau,
                rival_power=empirical_rival_power,
            )
        for j_value in j_values:
            m_value, q_value, iterations = mean_field_fixed_point(
                masses=masses,
                j_value=j_value,
                beta=beta,
                h_truth=h_truth,
                h_rival=h_rival,
                h_ambiguous=h_ambiguous,
                initial_m=initial_m,
            )
            phase = classify_mean_field_phase(
                m_value,
                q_value,
                m_threshold=m_threshold,
                q_threshold=q_threshold,
            )
            rows.append(
                {
                    "N": float(n_value),
                    "J": float(j_value),
                    "w_T": float(masses["w_T"]),
                    "w_R": float(masses["w_R"]),
                    "w_0": float(masses["w_0"]),
                    "m_star": m_value,
                    "q_star": q_value,
                    "phase": phase,
                    "iterations": iterations,
                }
            )
    return rows


def simulate_abstract_finite_agents(
    n_value: int,
    *,
    masses: dict[str, Any],
    trials: int,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    kappa: int,
    j_msg: float,
    h_truth: float,
    h_rival: float,
    h_ambiguous: float,
    sigma_truth: float,
    sigma_rival: float,
    sigma_ambiguous: float,
) -> dict[str, Any]:
    weights = np.array([masses["w_T"], masses["w_R"], masses["w_0"]], dtype=float)
    categories = rng.choice(3, size=(trials, n_value), p=weights)
    h_values = np.zeros((trials, n_value), dtype=float)
    truth_mask = categories == 0
    rival_mask = categories == 1
    ambiguous_mask = categories == 2
    h_values[truth_mask] = rng.normal(h_truth, sigma_truth, size=int(np.sum(truth_mask)))
    h_values[rival_mask] = rng.normal(h_rival, sigma_rival, size=int(np.sum(rival_mask)))
    h_values[ambiguous_mask] = rng.normal(h_ambiguous, sigma_ambiguous, size=int(np.sum(ambiguous_mask)))

    ell = h_values.copy()
    trial_index = np.arange(trials)
    for _ in range(kappa * n_value):
        speaker = rng.integers(0, n_value, size=trials)
        listener = rng.integers(0, n_value - 1, size=trials)
        listener = listener + (listener >= speaker)
        speaker_pref = np.tanh(beta * ell[trial_index, speaker])
        message = np.where(rng.random(trials) < (1.0 + speaker_pref) / 2.0, 1.0, -1.0)
        anchored_state = h_values[trial_index, listener] + j_msg * message
        ell[trial_index, listener] = (1.0 - alpha) * ell[trial_index, listener] + alpha * anchored_state

    truth_share = np.mean(ell > 0.0, axis=1)
    rival_share = np.mean(ell < 0.0, axis=1)
    endpoints = endpoint_from_vote_shares(truth_share, rival_share)
    counts = Counter(str(endpoint) for endpoint in endpoints)
    result: dict[str, Any] = {
        "N": int(n_value),
        "trials": int(trials),
        "J_msg": float(j_msg),
        "mean_v_T": float(np.mean(truth_share)),
        "mean_v_R": float(np.mean(rival_share)),
        "sd_v_T": float(np.std(truth_share)),
        "sd_v_R": float(np.std(rival_share)),
        "mean_type_T": float(np.mean(truth_mask)),
        "mean_type_R": float(np.mean(rival_mask)),
        "mean_type_0": float(np.mean(ambiguous_mask)),
        "mean_h": float(np.mean(h_values)),
        "sd_h": float(np.std(h_values)),
    }
    for endpoint in ENDPOINT_ORDER:
        result[endpoint] = float(counts[endpoint] / trials)
    result["modal_endpoint"] = max(
        ENDPOINT_ORDER,
        key=lambda endpoint: (float(result[endpoint]), -ENDPOINT_ORDER.index(endpoint)),
    )
    return result


def finite_agent_rows(
    mixture_rows: list[dict[str, Any]],
    *,
    trials: int,
    seed: int,
    alpha: float,
    beta: float,
    kappa: int,
    j_msg: float,
    h_truth: float,
    h_rival: float,
    h_ambiguous: float,
    sigma_truth: float,
    sigma_rival: float,
    sigma_ambiguous: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [
        simulate_abstract_finite_agents(
            int(masses["N"]),
            masses=masses,
            trials=trials,
            rng=rng,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            j_msg=j_msg,
            h_truth=h_truth,
            h_rival=h_rival,
            h_ambiguous=h_ambiguous,
            sigma_truth=sigma_truth,
            sigma_rival=sigma_rival,
            sigma_ambiguous=sigma_ambiguous,
        )
        for masses in mixture_rows
    ]


def simulate_crop_set(
    h_values: np.ndarray,
    *,
    j_value: float,
    trials: int,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    kappa: int,
) -> dict[str, float]:
    n_value = int(h_values.size)
    if n_value < 2:
        raise ValueError("Need at least two agents for pairwise simulation.")
    ell = np.tile(h_values.astype(float), (trials, 1))
    rows = np.arange(trials)
    for _ in range(kappa * n_value):
        speaker = rng.integers(0, n_value, size=trials)
        listener = rng.integers(0, n_value - 1, size=trials)
        listener = listener + (listener >= speaker)
        speaker_pref = np.tanh(beta * ell[rows, speaker])
        message = np.where(rng.random(trials) < (1.0 + speaker_pref) / 2.0, 1.0, -1.0)
        anchored_target = h_values[listener] + j_value * message
        ell[rows, listener] = (1.0 - alpha) * ell[rows, listener] + alpha * anchored_target

    truth_share = np.mean(ell > 0.0, axis=1)
    endpoints = endpoint_from_truth_share(truth_share)
    counts = Counter(str(value) for value in endpoints)
    result = {
        "trials": trials,
        "mean_truth_vote_share": float(np.mean(truth_share)),
        "mean_rival_vote_share": float(1.0 - np.mean(truth_share)),
    }
    for endpoint in ENDPOINT_ORDER:
        result[endpoint] = float(counts[endpoint] / trials)
    return result


def simulate_phase_rows(
    crop_sets: list[dict[str, Any]],
    *,
    n_values: list[int],
    j_values: list[float],
    trials: int,
    seed: int,
    alpha: float,
    beta: float,
    kappa: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for crop_set in crop_sets:
        grouped[int(crop_set["N"])].append(crop_set)

    for n_value in n_values:
        sets = grouped.get(n_value, [])
        if not sets:
            continue
        for j_value in j_values:
            total_trials = 0
            truth_sum = 0.0
            endpoint_counts = Counter()
            for crop_set in sets:
                h_values = np.array(crop_set["h_values"], dtype=float)
                simulated = simulate_crop_set(
                    h_values,
                    j_value=j_value,
                    trials=trials,
                    rng=rng,
                    alpha=alpha,
                    beta=beta,
                    kappa=kappa,
                )
                total_trials += trials
                truth_sum += float(simulated["mean_truth_vote_share"]) * trials
                for endpoint in ENDPOINT_ORDER:
                    endpoint_counts[endpoint] += float(simulated[endpoint]) * trials
            endpoint_rates = {
                endpoint: endpoint_counts[endpoint] / total_trials if total_trials else 0.0
                for endpoint in ENDPOINT_ORDER
            }
            modal_endpoint = max(
                ENDPOINT_ORDER,
                key=lambda endpoint: (endpoint_rates[endpoint], -ENDPOINT_ORDER.index(endpoint)),
            )
            rows.append(
                {
                    "N": n_value,
                    "J": float(j_value),
                    "n_seed_sets": len(sets),
                    "trials_per_seed": trials,
                    "total_trials": total_trials,
                    "mean_truth_vote_share": truth_sum / total_trials if total_trials else 0.0,
                    "mean_rival_vote_share": 1.0 - truth_sum / total_trials if total_trials else 0.0,
                    **endpoint_rates,
                    "modal_endpoint": modal_endpoint,
                    "modal_fraction": endpoint_rates[modal_endpoint],
                }
            )
    return rows


def load_empirical_endpoint_rows(
    run_root: Path,
    *,
    condition: str,
    n_values: list[int],
) -> list[dict[str, Any]]:
    return base.load_endpoint_regime_summary(run_root, condition=condition, n_values=n_values)


def rows_at_j_star(phase_rows: list[dict[str, Any]], j_star: float) -> list[dict[str, Any]]:
    if not phase_rows:
        return []
    available = sorted({float(row["J"]) for row in phase_rows})
    chosen = min(available, key=lambda value: abs(value - j_star))
    return sorted(
        [row for row in phase_rows if abs(float(row["J"]) - chosen) < 1e-9],
        key=lambda row: int(row["N"]),
    )


def draw_panel_b(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    n_values: list[int],
    *,
    theta: float,
) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No mixture rows", ha="center", va="center", color=base.GRAY)
        base.panel_title(ax, r"B. Pre-social field masses by $h_i$")
        return
    x = np.array([int(row["N"]) for row in rows], dtype=int)
    series = {
        "T": np.array([float(row["cutoff_w_T"]) for row in rows], dtype=float),
        "R": np.array([float(row["cutoff_w_R"]) for row in rows], dtype=float),
        "0": np.array([float(row["cutoff_w_0"]) for row in rows], dtype=float),
    }
    markers = {"T": "o", "R": "s", "0": "^"}
    linestyles = {"T": "-", "R": "-", "0": (0, (2.0, 1.4))}
    for key in FIELD_TYPE_ORDER:
        ax.plot(
            x,
            series[key],
            color=FIELD_TYPE_COLORS[key],
            marker=markers[key],
            markerfacecolor=base.WHITE,
            markeredgecolor=FIELD_TYPE_COLORS[key],
            markeredgewidth=1.05,
            linestyle=linestyles[key],
            linewidth=1.55,
            label=FIELD_TYPE_LABELS[key],
        )
    base.setup_log_n_axis(ax, n_values)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Mixture mass")
    base.panel_title(ax, r"B. Pre-social field masses by $h_i$")
    base.style_axis(ax)
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=7.2,
        handlelength=1.45,
        borderaxespad=0.12,
    )


def draw_panel_c(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    n_values: list[int],
    j_star: float,
) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No phase rows", ha="center", va="center", color=base.GRAY)
        base.panel_title(ax, r"D. Post-social phases over $J$")
        return
    n_grid = sorted({float(row["N"]) for row in rows})
    j_grid = sorted({float(row["J"]) for row in rows})
    endpoint_to_index = {endpoint: index for index, endpoint in enumerate(ENDPOINT_ORDER)}
    phase_array = np.full((len(j_grid), len(n_grid)), endpoint_to_index["fragmentation"], dtype=float)
    row_lookup = {(round(float(row["N"]), 10), round(float(row["J"]), 10)): row for row in rows}
    for j_idx, j_value in enumerate(j_grid):
        for n_idx, n_value in enumerate(n_grid):
            row = row_lookup.get((round(float(n_value), 10), round(float(j_value), 10)))
            if row is not None:
                phase_array[j_idx, n_idx] = endpoint_to_index[str(row["phase"])]
    log_centers = np.log(np.array(n_grid, dtype=float))
    log_step = float(np.median(np.diff(log_centers))) if len(log_centers) > 1 else 0.1
    n_edges = np.exp(np.concatenate(([log_centers[0] - log_step / 2.0], 0.5 * (log_centers[:-1] + log_centers[1:]), [log_centers[-1] + log_step / 2.0])))
    j_centers = np.array(j_grid, dtype=float)
    j_step = float(np.median(np.diff(j_centers))) if len(j_centers) > 1 else 0.1
    j_edges = np.concatenate(([j_centers[0] - j_step / 2.0], 0.5 * (j_centers[:-1] + j_centers[1:]), [j_centers[-1] + j_step / 2.0]))
    phase_cmap = ListedColormap([ENDPOINT_COLORS[endpoint] for endpoint in ENDPOINT_ORDER])
    norm = matplotlib.colors.BoundaryNorm(np.arange(len(ENDPOINT_ORDER) + 1) - 0.5, phase_cmap.N)
    ax.pcolormesh(
        n_edges,
        j_edges,
        phase_array,
        cmap=phase_cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
        rasterized=True,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n_value) for n_value in n_values])
    y_ticks = np.arange(math.ceil(min(j_grid) * 2.0) / 2.0, max(j_grid) + 0.001, 0.5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{value:g}" for value in y_ticks])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Social coupling J")
    chosen_j = min(j_grid, key=lambda value: abs(value - j_star))
    ax.axhline(chosen_j, color=base.INK, linewidth=1.2)
    label_offset = 0.04 * (max(j_grid) - min(j_grid))
    label_above = chosen_j < (max(j_grid) - 3.0 * label_offset)
    ax.text(
        5.0,
        chosen_j + label_offset if label_above else chosen_j - label_offset,
        r"$J^*=J_{\mathrm{msg}}$ in C",
        ha="left",
        va="bottom" if label_above else "top",
        fontsize=6.7,
        color=base.INK,
    )
    ax.grid(color=base.WHITE, linewidth=0.25, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(base.INK)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", colors=base.INK, direction="out", top=False, right=False)
    base.panel_title(ax, r"D. Post-social phases over $J$")


def draw_stacked_bars(
    ax: plt.Axes,
    x: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    width: float,
    alpha: float = 1.0,
) -> None:
    bottom = np.zeros(len(rows), dtype=float)
    for endpoint in ENDPOINT_ORDER:
        values = np.array([float(row.get(endpoint, 0.0)) for row in rows], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=width,
            color=ENDPOINT_COLORS[endpoint],
            edgecolor=base.WHITE,
            linewidth=0.45,
            alpha=alpha,
        )
        bottom += values


def draw_panel_e(
    ax: plt.Axes,
    sim_rows: list[dict[str, Any]],
    n_values: list[int],
) -> None:
    sim_by_n = {int(row["N"]): row for row in sim_rows}
    plotted_ns = [n_value for n_value in n_values if n_value in sim_by_n]
    if not plotted_ns:
        ax.text(0.5, 0.5, "No endpoint rows", ha="center", va="center", color=base.GRAY)
        base.panel_title(ax, r"C. Post-exchange endpoints at $J^*$")
        return
    x = np.arange(len(plotted_ns), dtype=float)
    sim_rows_ordered = [sim_by_n[n_value] for n_value in plotted_ns]
    draw_stacked_bars(ax, x, sim_rows_ordered, width=0.62, alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n_value) for n_value in plotted_ns])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Fraction of runs")
    base.panel_title(ax, r"C. Post-exchange endpoints at $J^*$")
    base.style_axis(ax)


def draw_figure(
    *,
    stem: str,
    panel_fields: list[dict[str, Any]],
    mixture_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    n_values: list[int],
    j_star: float,
    theta: float,
) -> list[Path]:
    base.setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.1, 2.85), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.35, 1.0, 1.06, 1.22],
        left=0.055,
        right=0.992,
        top=0.84,
        bottom=0.300,
        wspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[0, 3])

    base.draw_field_map(ax_a, panel_fields, theta)
    base.panel_title(ax_a, r"A. Empirical example of the crop field $h_i$")
    draw_panel_b(
        ax_b,
        mixture_rows,
        n_values,
        theta=theta,
    )
    draw_panel_e(ax_c, finite_rows, n_values)
    draw_panel_c(ax_d, phase_rows, n_values=n_values, j_star=j_star)

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    divider_x = 0.5 * (pos_a.x1 + pos_b.x0)
    fig.add_artist(
        matplotlib.lines.Line2D(
            [divider_x, divider_x],
            [0.300, 0.875],
            transform=fig.transFigure,
            color=base.LIGHT_GRAY,
            linewidth=1.0,
            alpha=0.95,
        )
    )

    handles = [Patch(facecolor=ENDPOINT_COLORS[key], edgecolor=base.WHITE, label=ENDPOINT_LABELS[key]) for key in ENDPOINT_ORDER]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.800, 0.020),
        ncol=4,
        fontsize=7.8,
        handlelength=1.0,
        columnspacing=0.75,
    )
    paths = [
        FIGURE_DIR / f"{stem}.png",
        FIGURE_DIR / f"{stem}.pdf",
        FIGURE_DIR / f"{stem}.svg",
    ]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    stem: str,
    mixture_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    figure_paths: list[Path],
    metadata: dict[str, Any],
) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mixture_path = DATA_DIR / f"{stem}_abstract_field_mixture.csv"
    write_rows(
        mixture_path,
        mixture_rows,
        [
            "N",
            "w_T",
            "w_R",
            "w_0",
            "cutoff_w_T",
            "cutoff_w_R",
            "cutoff_w_0",
            "theta",
            "raw_w_T",
            "raw_w_R",
            "rescale",
        ],
    )
    phase_path = DATA_DIR / f"{stem}_mean_field_phase_diagram.csv"
    write_rows(
        phase_path,
        phase_rows,
        [
            "N",
            "J",
            "w_T",
            "w_R",
            "w_0",
            "m_star",
            "q_star",
            "phase",
            "iterations",
        ],
    )
    finite_path = DATA_DIR / f"{stem}_finite_agent_votes.csv"
    write_rows(
        finite_path,
        finite_rows,
        [
            "N",
            "trials",
            "J_msg",
            "mean_v_T",
            "mean_v_R",
            "sd_v_T",
            "sd_v_R",
            "mean_type_T",
            "mean_type_R",
            "mean_type_0",
            "mean_h",
            "sd_h",
            "correct_consensus",
            "wrong_consensus",
            "polarization",
            "fragmentation",
            "modal_endpoint",
        ],
    )
    endpoint_path = DATA_DIR / f"{stem}_finite_agent_endpoint_regimes.csv"
    write_rows(
        endpoint_path,
        finite_rows,
        ["N", "trials", "J_msg", "correct_consensus", "wrong_consensus", "polarization", "fragmentation", "modal_endpoint"],
    )
    summary_path = DATA_DIR / f"{stem}_summary.json"
    payload = {
        **metadata,
        "outputs": {
            "abstract_field_mixture": str(mixture_path),
            "mean_field_phase_diagram": str(phase_path),
            "finite_agent_votes": str(finite_path),
            "finite_agent_endpoint_regimes": str(endpoint_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return [mixture_path, phase_path, finite_path, endpoint_path, summary_path, *figure_paths]


def split_csv_items(text: str | None) -> set[str] | None:
    if text is None or not text.strip():
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-input", type=Path, action="append", required=True)
    parser.add_argument("--mechanism-run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--mechanism-condition", default="all_gpt_4o")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--panel-target-country", default="France")
    parser.add_argument("--rival-country", default=None, help="Optional fixed rival. Leave unset for strongest measured rival.")
    parser.add_argument("--simulation-target-country", default=None, help="Optional target filter for simulations.")
    parser.add_argument("--seed-filter", default=None, help="Comma-separated seed dirs, e.g. seed_0003,seed_0010.")
    parser.add_argument("--max-seeds-per-n", type=int, default=None)
    parser.add_argument("--crop-set-mode", choices=("exact", "nested_prefix"), default="exact")
    parser.add_argument("--n-values", default=",".join(str(value) for value in DEFAULT_N_VALUES))
    parser.add_argument("--j-values", default=",".join(f"{value:g}" for value in default_j_grid()))
    parser.add_argument("--j-star", type=float, default=1.80)
    parser.add_argument("--j-msg", type=float, default=None, help="Finite-agent message coupling. Defaults to --j-star.")
    parser.add_argument("--phase-n-grid-size", type=int, default=300)
    parser.add_argument("--phase-j-grid-size", type=int, default=300)
    parser.add_argument("--phase-j-max", type=float, default=2.10)
    parser.add_argument("--phase-trials", type=int, default=300)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--lambda-smoothing", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--theta", type=float, default=math.log(2.0))
    parser.add_argument("--beta", type=float, default=1.30)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--kappa", type=int, default=24)
    parser.add_argument("--h-truth", type=float, default=1.20)
    parser.add_argument("--h-rival", type=float, default=-1.20)
    parser.add_argument("--h-ambiguous", type=float, default=0.0)
    parser.add_argument("--sigma-truth", type=float, default=0.20)
    parser.add_argument("--sigma-rival", type=float, default=0.20)
    parser.add_argument("--sigma-ambiguous", type=float, default=0.35)
    parser.add_argument("--mixture-mode", choices=("stylized", "empirical_anchor"), default="stylized")
    parser.add_argument("--max-informative-mass", type=float, default=0.55)
    parser.add_argument("--empirical-anchor-truth-tau", type=float, default=5.2)
    parser.add_argument("--empirical-anchor-rival-tau", type=float, default=13.2)
    parser.add_argument("--empirical-anchor-rival-power", type=float, default=0.86)
    parser.add_argument("--empirical-anchor-truth-mass", type=float, default=None)
    parser.add_argument("--empirical-anchor-rival-mass", type=float, default=None)
    parser.add_argument("--empirical-anchor-ambiguous-mass", type=float, default=None)
    parser.add_argument("--mean-field-initial-m", type=float, default=None)
    parser.add_argument("--mean-field-m-threshold", type=float, default=0.70)
    parser.add_argument("--mean-field-q-threshold", type=float, default=0.45)
    parser.add_argument("--allow-missing-mechanism-crops", action="store_true")
    parser.add_argument("--out-stem", default="flag_game_mechanism1_measured_field_final")
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    phase_n_grid = np.exp(np.linspace(np.log(min(n_values)), np.log(max(n_values)), args.phase_n_grid_size))
    phase_j_values = [float(value) for value in np.linspace(0.0, args.phase_j_max, args.phase_j_grid_size)]
    probe_rows = base.load_probe_inputs(args.probe_input)
    fields = base.estimate_crop_fields(
        probe_rows,
        smoothing_lambda=args.lambda_smoothing,
        epsilon=args.epsilon,
        theta=args.theta,
        rival_country=args.rival_country,
    )
    if not fields:
        raise SystemExit("No measured crop fields could be estimated.")
    panel_fields = [
        row
        for row in fields
        if row["model"] == args.model and base.country_key(row["target"]) == base.country_key(args.panel_target_country)
    ]
    if not panel_fields:
        raise SystemExit(f"No panel-A fields found for model={args.model!r} target={args.panel_target_country!r}.")

    raw_anchor_masses = empirical_anchor_masses(panel_fields, theta=args.theta) if args.mixture_mode == "empirical_anchor" else None
    anchor_masses = raw_anchor_masses
    if anchor_masses is not None:
        anchor_masses = override_anchor_masses(
            anchor_masses,
            truth_mass=args.empirical_anchor_truth_mass,
            rival_mass=args.empirical_anchor_rival_mass,
            ambiguous_mass=args.empirical_anchor_ambiguous_mass,
        )
    mixture_rows = abstract_mixture_rows(
        n_values,
        mode=args.mixture_mode,
        max_informative_mass=args.max_informative_mass,
        anchor_masses=anchor_masses,
        empirical_truth_tau=args.empirical_anchor_truth_tau,
        empirical_rival_tau=args.empirical_anchor_rival_tau,
        empirical_rival_power=args.empirical_anchor_rival_power,
    )
    for row in mixture_rows:
        row.update(
            cutoff_field_masses(
                row,
                theta=args.theta,
                h_truth=args.h_truth,
                h_rival=args.h_rival,
                h_ambiguous=args.h_ambiguous,
                sigma_truth=args.sigma_truth,
                sigma_rival=args.sigma_rival,
                sigma_ambiguous=args.sigma_ambiguous,
            )
        )
        row["theta"] = args.theta
    phase_rows = mean_field_phase_rows(
        phase_n_grid,
        mixture_mode=args.mixture_mode,
        j_values=phase_j_values,
        max_informative_mass=args.max_informative_mass,
        anchor_masses=anchor_masses,
        empirical_truth_tau=args.empirical_anchor_truth_tau,
        empirical_rival_tau=args.empirical_anchor_rival_tau,
        empirical_rival_power=args.empirical_anchor_rival_power,
        beta=args.beta,
        h_truth=args.h_truth,
        h_rival=args.h_rival,
        h_ambiguous=args.h_ambiguous,
        initial_m=args.mean_field_initial_m,
        m_threshold=args.mean_field_m_threshold,
        q_threshold=args.mean_field_q_threshold,
    )
    j_msg = args.j_msg if args.j_msg is not None else args.j_star
    finite_rows = finite_agent_rows(
        mixture_rows,
        trials=args.phase_trials,
        seed=args.seed + 101,
        alpha=args.alpha,
        beta=args.beta,
        kappa=args.kappa,
        j_msg=j_msg,
        h_truth=args.h_truth,
        h_rival=args.h_rival,
        h_ambiguous=args.h_ambiguous,
        sigma_truth=args.sigma_truth,
        sigma_rival=args.sigma_rival,
        sigma_ambiguous=args.sigma_ambiguous,
    )
    figure_paths = draw_figure(
        stem=args.out_stem,
        panel_fields=panel_fields,
        mixture_rows=mixture_rows,
        phase_rows=phase_rows,
        finite_rows=finite_rows,
        n_values=n_values,
        j_star=args.j_star,
        theta=args.theta,
    )
    output_paths = write_outputs(
        stem=args.out_stem,
        mixture_rows=mixture_rows,
        phase_rows=phase_rows,
        finite_rows=finite_rows,
        figure_paths=figure_paths,
        metadata={
            "truth_positive_sign_convention": "h_i = log((p_i(T)+epsilon)/(p_i(R)+epsilon))",
            "empirical_panel_role": (
                "Panel A motivates the signed-field abstraction; Panels B-D use abstract random fields "
                "without country identities, flag images, or LLM outputs."
            ),
            "probe_inputs": [str(path) for path in args.probe_input],
            "model": args.model,
            "panel_target_country": args.panel_target_country,
            "rival_country": args.rival_country,
            "n_values": n_values,
            "phase_n_grid_size": args.phase_n_grid_size,
            "phase_j_grid_size": args.phase_j_grid_size,
            "phase_j_min": 0.0,
            "phase_j_max": args.phase_j_max,
            "mixture_mode": args.mixture_mode,
            "raw_empirical_anchor_masses": raw_anchor_masses,
            "empirical_anchor_masses": anchor_masses,
            "empirical_anchor_truth_mass_override": args.empirical_anchor_truth_mass,
            "empirical_anchor_rival_mass_override": args.empirical_anchor_rival_mass,
            "empirical_anchor_ambiguous_mass_override": args.empirical_anchor_ambiguous_mass,
            "empirical_anchor_truth_tau": args.empirical_anchor_truth_tau,
            "empirical_anchor_rival_tau": args.empirical_anchor_rival_tau,
            "empirical_anchor_rival_power": args.empirical_anchor_rival_power,
            "j_star": args.j_star,
            "j_msg": j_msg,
            "finite_agent_trials_per_N": args.phase_trials,
            "alpha": args.alpha,
            "beta": args.beta,
            "kappa": args.kappa,
            "h_truth": args.h_truth,
            "h_rival": args.h_rival,
            "h_ambiguous": args.h_ambiguous,
            "sigma_truth": args.sigma_truth,
            "sigma_rival": args.sigma_rival,
            "sigma_ambiguous": args.sigma_ambiguous,
            "max_informative_mass": args.max_informative_mass,
            "mean_field_m_threshold": args.mean_field_m_threshold,
            "mean_field_q_threshold": args.mean_field_q_threshold,
        },
    )
    for path in output_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
