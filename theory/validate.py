"""Recompute and validate the accepted model using only this folder's inputs.

Runs independent birth-death balance, absorbing fixation, and neutral-copying
checks; compares the full stored grid and the reviewed run-mixture values.
Writes validation/model_validation.json. No simulation or parameter fitting.
"""

from collections import Counter, defaultdict
import csv
from math import comb, exp
from pathlib import Path
import hashlib
import json

import numpy as np

import model


HERE = Path(__file__).resolve().parent
TOL = 1e-10


def read_csv(name):
    with (HERE / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def independently_solved_fixation(N, h0):
    """Solve the absorbing birth-death backward equations, then binomial-average."""
    up = exp(h0)
    matrix = np.diag(np.full(N - 1, up + 1))
    if N > 2:
        matrix += np.diag(np.full(N - 2, -up), 1)
        matrix += np.diag(np.full(N - 2, -1.0), -1)
    right = np.zeros(N - 1)
    right[-1] = up
    fixation = np.concatenate(([0.0], np.linalg.solve(matrix, right), [1.0]))
    initial = np.array([comb(N, k) / 2 ** N for k in range(N + 1)])
    return float(initial @ fixation)


def structural_checks():
    errors = dict(stationary_mass=0.0, normalized_detailed_balance=0.0,
                  neutral_conditional_mean=0.0, no_anchor_fixation=0.0,
                  neutral_symmetric_mean=0.0, one_anchor_fixation=0.0)
    checked = 0
    for N in model.POPULATIONS:
        for h0 in (model.H0, 0.0):
            numerical = independently_solved_fixation(N, h0)
            errors["no_anchor_fixation"] = max(errors["no_anchor_fixation"], abs(numerical - model.no_anchor_truth_fixation(N, h0)))
        for t in range(1, N):
            for r in range(1, N - t + 1):
                truth, pi = model.stationary_truth_counts(N, t, r)
                free = N - t - r
                errors["stationary_mass"] = max(errors["stationary_mass"], float(abs(pi.sum() - 1)))
                assert np.all(pi >= 0)
                if free:
                    n = np.arange(free)
                    up = exp(model.H0) * (free - n) * (t + n)
                    down = (n + 1) * (r + free - n - 1)
                    errors["normalized_detailed_balance"] = max(
                        errors["normalized_detailed_balance"],
                        float(np.max(np.abs(pi[:-1] * up - pi[1:] * down))) / (N * N),
                    )
                neutral = model.conditional_mean(N, t, r, 0.0)
                errors["neutral_conditional_mean"] = max(errors["neutral_conditional_mean"], abs(neutral - t / (t + r)))
                checked += 1
        for anchors in range(1, N + 1):
            errors["one_anchor_fixation"] = max(errors["one_anchor_fixation"],
                float(np.abs(model.conditional_endpoint(N, anchors, 0) - [1, 0, 0]).max()),
                float(np.abs(model.conditional_endpoint(N, 0, anchors) - [0, 1, 0]).max()))
        errors["neutral_symmetric_mean"] = max(errors["neutral_symmetric_mean"], abs(model.mean_truth_shares(N, 0.5, h0=0) - 0.5))
    assert max(errors.values()) < TOL, errors
    return {"both_anchor_compositions_checked": checked, "maximum_errors": errors}


def grid_checks():
    with np.load(HERE / "data/model_probabilities.npz") as grid:
        populations, shares = grid["N"].copy(), grid["share"].copy()
        stored = grid["probabilities"].copy()
        assert float(grid["reference_share"]) == model.REFERENCE_SHARE
        assert str(grid["method"]) == model.METHOD == "exact_binary_birth_death"
        assert float(grid["total_anchor_probability"]) == model.TOTAL_ANCHOR_PROBABILITY
        assert float(grid["h0"]) == model.H0
    assert tuple(populations) == model.POPULATIONS
    assert np.array_equal(shares, np.linspace(0, 1, 1001))
    assert stored.shape == (3, 7, 1001)
    assert tuple(populations) == (2, 4, 8, 16, 32, 64, 128)
    recomputed = np.stack([model.endpoint_probs(int(N), shares) for N in populations], axis=1)
    errors = {
        "stored_grid": float(np.max(np.abs(recomputed - stored))),
        "probability_mass": float(np.max(np.abs(recomputed.sum(axis=0) - 1))),
    }
    assert np.all(recomputed >= -TOL) and np.all(recomputed <= 1 + TOL)
    reference = read_csv("model_reference_probabilities.csv")
    errors["reference_slice"] = max(float(np.max(np.abs(
        model.endpoint_probs(int(row["N"]), model.REFERENCE_SHARE)
        - np.array([float(row[name]) for name in model.OUTCOMES])
    ))) for row in reference)
    errors["composition_mass"] = max(float(np.max(np.abs(
        model.composition_weights(int(N), [0, 0.35, 0.5, 1]).sum(axis=0) - 1
    ))) for N in populations)
    assert max(errors.values()) < TOL, errors
    fixed = [{"N": int(N), "mean_truth_share": model.mean_truth_shares(int(N), model.REFERENCE_SHARE),
              **dict(zip(model.OUTCOMES, map(float, model.endpoint_probs(int(N), model.REFERENCE_SHARE))))}
             for N in populations]
    return {"grid_shape": list(stored.shape), "maximum_errors": errors, "reference_slice": fixed}


def run_mixture_checks():
    runs = read_csv("model_run_reference.csv")
    grouped = defaultdict(list)
    assert len(runs) == 217 and len({row["run_id"] for row in runs}) == 217
    count_errors = []
    for row in runs:
        N = int(row["N"])
        truth, rival, valid = (int(row[key]) for key in ("truth_count", "rival_count", "valid_count"))
        assert truth + rival == valid == 50 * N
        count_errors.append(abs(rival / valid - float(row["rival_response_share"])))
        grouped[N].append(row)
    references = {int(row["N"]): row for row in read_csv("model_mixture_reference.csv")}
    max_per_run_error = 0.0
    max_mixture_error = 0.0
    comparisons = []
    for N, rows in sorted(grouped.items()):
        shares = np.array([float(row["rival_response_share"]) for row in rows])
        probabilities = model.endpoint_probs(N, shares)
        means = model.mean_truth_shares(N, shares)
        stored_probabilities = np.array([[float(row[f"prob_{name}"]) for row in rows] for name in model.OUTCOMES])
        stored_means = np.array([float(row["mean_truth_share"]) for row in rows])
        max_per_run_error = max(max_per_run_error, float(np.max(np.abs(probabilities - stored_probabilities))), float(np.max(np.abs(means - stored_means))))
        observed = np.array([float(row["observed_endpoint_accuracy"]) for row in rows])
        actual = {
            "N": N, "runs": len(rows), "mean_rival_response_share": float(shares.mean()),
            "model_mean_truth_share": float(means.mean()), "observed_mean_accuracy": float(observed.mean()),
            **{f"model_prob_{name}": float(probabilities[j].mean()) for j, name in enumerate(model.OUTCOMES)},
            **{f"observed_prob_{name}": sum(row["observed_outcome"] == name for row in rows) / len(rows) for name in model.EMPIRICAL_OUTCOMES},
        }
        max_mixture_error = max(max_mixture_error, max(abs(value - float(references[N][key])) for key, value in actual.items()))
        comparisons.append(actual)
    assert max(max_per_run_error, max_mixture_error, *count_errors) < TOL
    model_peak = max(comparisons, key=lambda row: row["model_mean_truth_share"])["N"]
    observed_peak = max(comparisons, key=lambda row: row["observed_mean_accuracy"])["N"]
    assert model_peak == observed_peak == 32
    return {
        "run_count": len(runs), "outcome_counts": dict(Counter(row["observed_outcome"] for row in runs)),
        "max_count_share_error": max(count_errors), "max_per_run_model_error": max_per_run_error,
        "max_run_mixture_error": max_mixture_error, "model_mean_truth_peak_N": model_peak,
        "observed_mean_accuracy_peak_N": observed_peak, "comparisons": comparisons,
        "interpretation": "Descriptive equal-weight averaging over each N's measured run-specific shares; retrospective rival selection and differing share distributions across N prevent interpreting the mixture peak as a fixed-distribution model prediction.",
    }


def empirical_data_checks():
    """Reconstruct each plotted share from the packaged 50-response crop counts."""
    crops = read_csv("crop_probe_counts.csv")
    empirical = read_csv("empirical_runs.csv")
    excluded = read_csv("excluded_runs.csv")
    endpoints = {row["run_id"]: row for row in read_csv("main_sweep_endpoints.csv")}
    references = {row["run_id"]: row for row in read_csv("model_run_reference.csv")}
    grouped = defaultdict(list)
    assert len(crops) == 9448 and len(empirical) == 217 and len(excluded) == 10
    for row in crops:
        assert int(row["B_valid"]) == 50
        assert int(row["truth_count"]) + int(row["rival_count"]) == 50
        assert row["outcome"] == endpoints[row["run_id"]]["outcome"]
        grouped[row["run_id"]].append(row)
    included_ids = {row["run_id"] for row in empirical}
    excluded_ids = {row["run_id"] for row in excluded}
    assert len(included_ids) == 217 and len(excluded_ids) == 10
    assert included_ids.isdisjoint(excluded_ids)
    assert included_ids | excluded_ids == set(grouped) == set(endpoints)
    assert included_ids == set(references)
    max_share_error = 0.0
    for row in [*empirical, *excluded]:
        crop_rows = grouped[row["run_id"]]
        N = int(endpoints[row["run_id"]]["N"])
        assert len(crop_rows) == N
        assert len({int(crop["agent_id"]) for crop in crop_rows}) == N
        totals = {key: sum(int(crop[key]) for crop in crop_rows) for key in ("truth_count", "rival_count", "B_valid")}
        assert int(row["truth_count"]) == totals["truth_count"]
        assert int(row["rival_count"]) == totals["rival_count"]
        assert int(row["valid_count"]) == totals["B_valid"]
        max_share_error = max(max_share_error, abs(float(row["rival_response_share"]) - totals["rival_count"] / totals["B_valid"]))
        if row["run_id"] in included_ids:
            reference = references[row["run_id"]]
            assert row["outcome"] == endpoints[row["run_id"]]["outcome"] == reference["observed_outcome"]
            assert abs(float(reference["observed_endpoint_accuracy"]) - float(endpoints[row["run_id"]]["accuracy"])) < TOL
            assert abs(float(reference["rival_response_share"]) - float(row["rival_response_share"])) < TOL
    example = json.loads((HERE / "data/yemen_example.json").read_text())
    yemen = grouped[example["run_id"]]
    assert len(yemen) == 128
    assert sum(int(row["rival_count"]) for row in yemen) == example["rival_count"] == 4192
    assert sum(int(row["B_valid"]) for row in yemen) == example["valid_count"] == 6400
    assert example["rival_response_share"] == 4192 / 6400 == 0.655
    panel_crops = read_csv("yemen_crops.csv")
    by_agent = {row["agent_id"]: row for row in yemen}
    assert len(panel_crops) == 128
    for row in panel_crops:
        original = by_agent[row["agent_id"]]
        for field in ("left", "top", "width", "height", "truth_count", "rival_count", "B_valid"):
            assert float(row[field]) == float(original[field])
        assert abs(float(row["rival_response_share"]) - int(original["rival_count"]) / 50) < TOL
    assert max_share_error < TOL
    return {"source_runs": len(grouped), "source_crop_placements": len(crops),
            "valid_responses_per_crop": 50, "included_runs": len(empirical), "excluded_runs": len(excluded),
            "max_pooled_share_error": max_share_error, "yemen_rival_responses": 4192,
            "yemen_valid_responses": 6400, "yemen_rival_response_share": 0.655,
            "exclusion_note": "The ten preserved exclusions were made under the earlier rectified-evidence measure; they do not indicate zero literal truth-response probability."}


def binary_classification_checks():
    """Check exhaustive fixed states, including exact 85/15 threshold ties."""
    checked = 0
    for N in sorted({*model.POPULATIONS, 20}):
        for truth in range(N + 1):
            expected = np.zeros(3)
            if 20 * truth >= 17 * N:
                expected[0] = 1
            elif 20 * truth <= 3 * N:
                expected[1] = 1
            else:
                expected[2] = 1
            actual = model.conditional_endpoint(N, truth, N - truth)
            assert np.array_equal(actual, expected)
            checked += 1
    assert model.OUTCOMES == ("correct_consensus", "wrong_consensus", "polarization")
    assert model.EMPIRICAL_OUTCOMES == (*model.OUTCOMES, "fragmentation")
    assert np.array_equal(model.conditional_endpoint(8, 6, 2), [0, 0, 1])
    assert np.array_equal(model.conditional_endpoint(8, 7, 1), [1, 0, 0])
    assert np.array_equal(model.conditional_endpoint(16, 13, 3), [0, 0, 1])
    return {"fixed_states_checked": checked, "consensus_threshold": 0.85,
            "polarization_truth_share_interval": "(0.15, 0.85)",
            "empirical_categories_unchanged": True}


def main():
    results = {
        "passed": True,
        "model_method": model.METHOD,
        "scope": "Exact discrete binary stationary/fixation calculation; no diffusion approximation. Empirical multicountry labels are preserved.",
        "model_sha256": hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest(),
        "binary_classification_checks": binary_classification_checks(),
        "parameters": {"total_anchor_probability": model.TOTAL_ANCHOR_PROBABILITY, "h0": model.H0,
                       "reference_share": model.REFERENCE_SHARE, "population_sizes": list(model.POPULATIONS)},
        "outcome_order": list(model.OUTCOMES), "tolerance": TOL,
        "method": "Exact stationary distributions for both anchors, fixation otherwise, and fair binary initialization without anchors. No fitting or simulation.",
        "empirical_data_checks": empirical_data_checks(),
        "structural_checks": structural_checks(), "grid_checks": grid_checks(), "run_mixture_checks": run_mixture_checks(),
        "input_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [HERE / "model.py", HERE / "validate.py", *sorted((HERE / "data").glob("*.csv")), HERE / "data/model_probabilities.npz", HERE / "data/yemen_example.json"]},
    }
    (HERE / "validation").mkdir(exist_ok=True)
    (HERE / "validation/model_validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": True, "grid_max_error": results["grid_checks"]["maximum_errors"]["stored_grid"],
                      "run_count": results["run_mixture_checks"]["run_count"], "output": "validation/model_validation.json"}, indent=2))


if __name__ == "__main__":
    main()
