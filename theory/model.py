"""Exact long-time probabilities for the accepted binary copying model.

Truth and rival anchors are fixed; remaining agents copy with relative
truth/rival acceptance rates exp(h0):1. Anchor counts are multinomial with
probabilities a_T=total_anchor_probability*(1-share),
a_R=total_anchor_probability*share, and a_0=1-total_anchor_probability.

Both-anchor populations use the exact stationary birth-death distribution.
One-anchor populations fixate at that anchor's label. Populations with no
anchors use fixation probabilities averaged over fair independent binary
initial states. These are long-time results, not finite-round predictions.

Run ``python model.py --recompute`` to regenerate the packaged model grid.
No empirical fitting, remote services, or files outside this folder are used.
"""

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
from math import exp, expm1, lgamma, log
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TOTAL_ANCHOR_PROBABILITY = 0.45
H0 = 0.3
REFERENCE_SHARE = 0.35
METHOD = "exact_binary_birth_death"
POPULATIONS = (2, 4, 8, 16, 32, 64, 128)
OUTCOMES = ("correct_consensus", "wrong_consensus", "polarization")
EMPIRICAL_OUTCOMES = (*OUTCOMES, "fragmentation")


def stationary_truth_counts(N, truth_anchors, rival_anchors, h0=H0):
    """Return (total truth counts, probabilities) when both anchor types exist."""
    t, r = int(truth_anchors), int(rival_anchors)
    if t < 1 or r < 1 or t + r > N:
        raise ValueError("Both anchor types must be present and fit within N.")
    free = N - t - r
    n = np.arange(free)
    log_ratios = (
        h0 + np.log(free - n) + np.log(t + n)
        - np.log(n + 1) - np.log(r + free - n - 1)
    )
    log_pi = np.concatenate(([0.0], np.cumsum(log_ratios)))
    pi = np.exp(log_pi - log_pi.max())
    pi /= pi.sum()
    return t + np.arange(free + 1), pi


def no_anchor_truth_fixation(N, h0=H0):
    """Analytically average binary fixation over Binomial(N, 1/2) initial truth."""
    if h0 == 0:
        return 0.5
    q = exp(-h0)
    return expm1(N * log((1 + q) / 2)) / expm1(-h0 * N)


def conditional_endpoint(N, truth_anchors, rival_anchors, h0=H0):
    """Three outcomes: 85% consensus at either label, otherwise polarization."""
    t, r = int(truth_anchors), int(rival_anchors)
    if N < 2 or t < 0 or r < 0 or t + r > N:
        raise ValueError("Require N >= 2 and valid nonnegative anchor counts.")
    if t == 0 and r == 0:
        fixation = no_anchor_truth_fixation(N, h0)
        return np.array([fixation, 1 - fixation, 0.0])
    if r == 0:
        return np.array([1.0, 0.0, 0.0])
    if t == 0:
        return np.array([0.0, 1.0, 0.0])
    truth_counts, pi = stationary_truth_counts(N, t, r, h0)
    classes = np.full(len(pi), 2)
    classes[truth_counts * 20 >= 17 * N] = 0
    classes[truth_counts * 20 <= 3 * N] = 1
    return np.bincount(classes, weights=pi, minlength=len(OUTCOMES))


def conditional_mean(N, truth_anchors, rival_anchors, h0=H0):
    """Mean final truth fraction, distinct from correct-consensus probability."""
    if truth_anchors == 0 or rival_anchors == 0:
        return float(conditional_endpoint(N, truth_anchors, rival_anchors, h0)[0])
    truth_counts, pi = stationary_truth_counts(N, truth_anchors, rival_anchors, h0)
    return float(pi @ truth_counts / N)


@lru_cache(maxsize=None)
def coefficients(N, h0=H0):
    """Anchor compositions, log multinomial coefficients, conditional outcomes."""
    compositions = [(t, r) for t in range(N + 1) for r in range(N - t + 1)]
    counts = np.array([(t, r, N - t - r) for t, r in compositions])
    log_coefficients = np.array([
        lgamma(N + 1) - lgamma(t + 1) - lgamma(r + 1) - lgamma(N - t - r + 1)
        for t, r in compositions
    ])
    endpoints = np.array([conditional_endpoint(N, t, r, h0) for t, r in compositions])
    return counts, log_coefficients, endpoints


@lru_cache(maxsize=None)
def conditional_means(N, h0=H0):
    counts, _, _ = coefficients(N, h0)
    return np.array([conditional_mean(N, int(t), int(r), h0) for t, r, _ in counts])


def _inputs(N, shares, total_anchor_probability, h0):
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2.")
    scalar = np.ndim(shares) == 0
    shares = np.atleast_1d(np.asarray(shares, dtype=float))
    if shares.ndim != 1 or not np.all(np.isfinite(shares)) or not np.all((shares >= 0) & (shares <= 1)):
        raise ValueError("Evidence shares must lie in [0, 1].")
    if not 0 <= total_anchor_probability <= 1 or not np.isfinite(h0):
        raise ValueError("Invalid anchor probability or social bias.")
    return int(N), shares, scalar


def composition_weights(N, shares, total_anchor_probability=TOTAL_ANCHOR_PROBABILITY, h0=H0):
    """Multinomial composition masses; columns correspond to evidence shares."""
    N, shares, _ = _inputs(N, shares, total_anchor_probability, h0)
    counts, log_coefficients, _ = coefficients(N, h0)
    # This operation order preserves the accepted figure's validated numerics.
    a_T = total_anchor_probability * (1 - shares)
    a_R = total_anchor_probability * shares
    probabilities = np.stack([a_T, a_R, 1 - a_T - a_R])
    log_weights = np.broadcast_to(log_coefficients[:, None], (len(counts), len(shares))).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for index in range(3):
            log_weights += np.where(
                counts[:, index, None] > 0,
                counts[:, index, None] * np.log(probabilities[index])[None, :],
                0.0,
            )
    return np.exp(log_weights)


def endpoint_probs(N, shares, total_anchor_probability=TOTAL_ANCHOR_PROBABILITY, h0=H0):
    """Exact probabilities in OUTCOMES order; scalar -> (3,), vector -> (3, m)."""
    N, shares, scalar = _inputs(N, shares, total_anchor_probability, h0)
    _, _, endpoints = coefficients(N, h0)
    result = np.empty((len(shares), len(OUTCOMES)))
    for start in range(0, len(shares), 200):
        selected = shares[start:start + 200]
        result[start:start + len(selected)] = composition_weights(N, selected, total_anchor_probability, h0).T @ endpoints
    return result[0] if scalar else result.T


def mean_truth_shares(N, shares, total_anchor_probability=TOTAL_ANCHOR_PROBABILITY, h0=H0):
    """Exact mean final truth fraction; scalar input returns a float."""
    N, shares, scalar = _inputs(N, shares, total_anchor_probability, h0)
    means = conditional_means(N, h0)
    result = np.empty(len(shares))
    for start in range(0, len(shares), 200):
        selected = shares[start:start + 200]
        result[start:start + len(selected)] = composition_weights(N, selected, total_anchor_probability, h0).T @ means
    return float(result[0]) if scalar else result


def _read_csv(name):
    with (HERE / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name, rows):
    with (HERE / "data" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recompute_run_references():
    """Evaluate selected run shares while preserving their empirical outcomes."""
    empirical = _read_csv("empirical_runs.csv")
    endpoints = {row["run_id"]: row for row in _read_csv("main_sweep_endpoints.csv")}
    crop_metadata = {row["run_id"]: row for row in _read_csv("crop_probe_counts.csv")}
    grouped = defaultdict(list)
    for row in empirical:
        grouped[int(row["N"])].append(row)
    per_run, mixtures = [], []
    for N, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (float(row["rival_response_share"]), row["run_id"]))
        shares = np.array([float(row["rival_response_share"]) for row in rows])
        probabilities = endpoint_probs(N, shares)
        means = mean_truth_shares(N, shares)
        observed = np.array([float(endpoints[row["run_id"]]["accuracy"]) for row in rows])
        for index, row in enumerate(rows):
            metadata = crop_metadata[row["run_id"]]
            per_run.append({
                "run_id": row["run_id"], "N": N, "truth": metadata["truth"], "rival": metadata["rival"],
                "rival_response_share": float(shares[index]),
                "a_T": TOTAL_ANCHOR_PROBABILITY * (1 - float(shares[index])),
                "a_R": TOTAL_ANCHOR_PROBABILITY * float(shares[index]),
                "mean_truth_share": float(means[index]),
                **{f"prob_{name}": float(probabilities[j, index]) for j, name in enumerate(OUTCOMES)},
                "observed_endpoint_accuracy": float(observed[index]), "observed_outcome": row["outcome"],
                **{key: int(row[key]) for key in ("truth_count", "rival_count", "valid_count")},
            })
        mixtures.append({
            "N": N, "runs": len(rows), "mean_rival_response_share": float(shares.mean()),
            "model_mean_truth_share": float(means.mean()), "observed_mean_accuracy": float(observed.mean()),
            **{f"model_prob_{name}": float(probabilities[j].mean()) for j, name in enumerate(OUTCOMES)},
            **{f"observed_prob_{name}": sum(row["outcome"] == name for row in rows) / len(rows) for name in EMPIRICAL_OUTCOMES},
        })
    _write_csv("model_run_reference.csv", per_run)
    _write_csv("model_mixture_reference.csv", mixtures)


def recompute():
    """Regenerate the accepted 0.001-share grid and panel-b reference values."""
    target = HERE / "data"
    target.mkdir(exist_ok=True)
    shares = np.linspace(0, 1, 1001)
    probabilities = np.stack([endpoint_probs(N, shares) for N in POPULATIONS], axis=1)
    np.savez_compressed(
        target / "model_probabilities.npz", N=POPULATIONS, share=shares,
        probabilities=probabilities, reference_share=REFERENCE_SHARE,
        method=METHOD, total_anchor_probability=TOTAL_ANCHOR_PROBABILITY, h0=H0,
    )
    with (target / "model_reference_probabilities.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["N", *OUTCOMES])
        writer.writeheader()
        writer.writerows({"N": N, **dict(zip(OUTCOMES, endpoint_probs(N, REFERENCE_SHARE)))} for N in POPULATIONS)
    recompute_run_references()
    print("Recomputed model grid, reference slice, and run-mixture references; empirical outcomes unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="regenerate the accepted model grid and reference slice")
    args = parser.parse_args()
    if args.recompute:
        recompute()
    else:
        parser.print_help()
