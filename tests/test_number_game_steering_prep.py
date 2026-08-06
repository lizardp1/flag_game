import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import run_number_game_steering_prep as steering


class SteeringPrepTests(unittest.TestCase):
    def test_sign_calibration_flips_negative_raw_slope(self):
        rows = [
            {
                "layer": 28,
                "alpha": -1.0,
                "mean_social_minus_private_number_logprob_margin": 2.0,
            },
            {
                "layer": 28,
                "alpha": 0.0,
                "mean_social_minus_private_number_logprob_margin": 0.0,
            },
            {
                "layer": 28,
                "alpha": 1.0,
                "mean_social_minus_private_number_logprob_margin": -2.0,
            },
        ]
        sign_rows = steering.calibrate_steering_signs(rows)
        self.assertEqual(sign_rows[0]["empirical_social_sign"], -1)
        self.assertEqual(sign_rows[0]["raw_positive_alpha_effect"], "more_private")
        self.assertEqual(rows[0]["calibrated_alpha"], 1.0)
        self.assertEqual(rows[2]["calibrated_alpha"], -1.0)

    def test_ood_memory_majority_ignores_truth(self):
        memory = ["12 | The number is even.", "12 | I think it is 12.", "7 | The number is prime."]
        self.assertEqual(steering.majority_non_target_memory_number(memory, 7), 12)

    def test_choice_category_tracks_social_private_and_clue(self):
        self.assertEqual(
            steering.choice_category(12, private_target=7, social_number=12, private_clue="the number is prime"),
            "social",
        )
        self.assertEqual(
            steering.choice_category(11, private_target=7, social_number=12, private_clue="the number is prime"),
            "other_clue_compatible",
        )
        self.assertEqual(
            steering.choice_category(14, private_target=7, social_number=12, private_clue="the number is prime"),
            "incompatible",
        )

    def test_parse_list_cell_accepts_runner_csv_repr(self):
        self.assertEqual(
            steering.parse_list_cell("['12 | The number is even.', '7 | The number is prime.']"),
            ["12 | The number is even.", "7 | The number is prime."],
        )

    def test_behavioral_summary_reports_social_choice_delta(self):
        rows = [
            {
                "dataset": "synthetic",
                "layer": 24,
                "calibrated_alpha": -10.0,
                "valid_rate": 1.0,
                "format_damage_rate": 0.0,
                "satisfies_private_clue_rate": 0.9,
                "social_choice_rate": 0.1,
                "private_target_choice_rate": 0.6,
            },
            {
                "dataset": "synthetic",
                "layer": 24,
                "calibrated_alpha": 0.0,
                "valid_rate": 1.0,
                "format_damage_rate": 0.0,
                "satisfies_private_clue_rate": 0.8,
                "social_choice_rate": 0.2,
                "private_target_choice_rate": 0.5,
            },
            {
                "dataset": "synthetic",
                "layer": 24,
                "calibrated_alpha": 10.0,
                "valid_rate": 1.0,
                "format_damage_rate": 0.0,
                "satisfies_private_clue_rate": 0.6,
                "social_choice_rate": 0.45,
                "private_target_choice_rate": 0.35,
            },
        ]
        summary = steering.behavioral_steering_effect_summary(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["positive_social_choice_delta"], 0.25)
        self.assertAlmostEqual(summary[0]["positive_satisfies_private_clue_delta"], -0.2)

    def test_choose_behavior_layer_prefers_social_delta_before_damage_tiebreak(self):
        rows = [{"layer": 20}, {"layer": 22}]
        behavioral_rows = [
            {"layer": 20, "best_social_choice_delta": 0.05, "best_format_damage_delta": 0.0},
            {"layer": 22, "best_social_choice_delta": 0.20, "best_format_damage_delta": 0.5},
        ]
        self.assertEqual(steering.choose_behavior_layer(rows, behavioral_rows), 22)

    def test_choose_behavior_layer_tiebreaks_on_lower_format_damage(self):
        rows = [{"layer": 20}, {"layer": 22}]
        behavioral_rows = [
            {"layer": 20, "best_social_choice_delta": 0.20, "best_format_damage_delta": 0.5},
            {"layer": 22, "best_social_choice_delta": 0.20, "best_format_damage_delta": 0.0},
        ]
        self.assertEqual(steering.choose_behavior_layer(rows, behavioral_rows), 22)

    def test_refresh_plots_from_csv_writes_layer_specific_generation_plots(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for layer in [20, 22]:
                for alpha in [0.0, 10.0]:
                    rows.append(
                        {
                            "dataset": "synthetic",
                            "layer": layer,
                            "calibrated_alpha": alpha,
                            "valid_rate": 1.0,
                            "format_damage_rate": 0.0,
                            "satisfies_private_clue_rate": 0.8,
                            "mean_base_completion_perplexity": 1.2,
                            "social_choice_rate": 0.1 + 0.01 * layer + 0.01 * alpha,
                            "private_target_choice_rate": 0.5,
                            "other_clue_compatible_rate": 0.3,
                            "incompatible_rate": 0.1,
                        }
                    )
            steering.write_csv(out_dir / "generation_side_effect_summary.csv", rows)
            steering.write_csv(
                out_dir / "behavioral_steering_effect_summary.csv",
                [
                    {"dataset": "synthetic", "layer": 20, "best_social_choice_delta": 0.01, "best_format_damage_delta": 0.0},
                    {"dataset": "synthetic", "layer": 22, "best_social_choice_delta": 0.20, "best_format_damage_delta": 0.0},
                ],
            )

            steering.refresh_plots_from_csv(out_dir)

            self.assertTrue((out_dir / "plots" / "generation_side_effects_layer_22.svg").exists())
            self.assertTrue((out_dir / "plots" / "generation_choice_composition_layer_22.svg").exists())
            self.assertTrue((out_dir / "plots" / "behavior_choice_composition_layer_22.svg").exists())

    def test_logprob_quantile_direction_points_from_low_to_high_margin(self):
        trials = []
        vectors = {}
        for index, margin in enumerate([-4.0, -3.0, 3.0, 4.0]):
            trial_id = f"trial_{index}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "pair_id": f"pair_{index}",
                    "case_id": f"case_{index}",
                    "variant": "social_memory",
                    "split": "train",
                    "social_minus_private_number_logprob_margin": margin,
                }
            )
            vectors[trial_id] = {1: np.array([margin, 0.0])}
        fit = steering.fit_direction(
            trials=trials,
            vectors=vectors,
            layer=1,
            fit_split="train",
            direction_method="logprob_quantile",
            direction_quantile=0.25,
            subspace_rank=1,
        )
        self.assertIsNotNone(fit)
        self.assertGreater(fit["direction"][0], 0.0)

    def test_mixed_ratio_memory_lines_use_private_and_social_counts(self):
        case = steering.SteeringCase(
            case_id="ratio_case",
            private_clue="the number is prime",
            private_target=7,
            private_reason="The number is prime.",
            social_number=12,
            social_reason="The number is even.",
        )
        m1_lines = steering.mixed_ratio_memory_lines(case, m=1, target_count=2, social_count=3)
        self.assertEqual(m1_lines.count("7"), 2)
        self.assertEqual(m1_lines.count("12"), 3)

        m3_lines = steering.mixed_ratio_memory_lines(case, m=3, target_count=1, social_count=1)
        self.assertEqual(len(m3_lines), 2)
        self.assertTrue(all(" | " in line for line in m3_lines))

    def test_interleaved_memory_sequence_does_not_block_all_one_kind_first(self):
        sequence = steering.interleaved_memory_sequence(target_count=4, social_count=4)
        self.assertEqual(sequence.count("target"), 4)
        self.assertEqual(sequence.count("social"), 4)
        self.assertIn("target", sequence[:3])
        self.assertIn("social", sequence[:3])

    def test_ratio_slope_direction_tracks_social_memory_fraction(self):
        trials = []
        vectors = {}
        for index, fraction in enumerate([0.0, 0.5, 1.0]):
            trial_id = f"ratio_{index}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "pair_id": "case_m1_ratio_total8",
                    "case_id": "case",
                    "variant": "ratio_memory",
                    "split": "train",
                    "social_memory_fraction": fraction,
                    "social_minus_private_number_logprob_margin": fraction,
                }
            )
            vectors[trial_id] = {1: np.array([fraction, 0.0])}

        fit = steering.fit_direction(
            trials=trials,
            vectors=vectors,
            layer=1,
            fit_split="train",
            direction_method="ratio_slope",
            direction_quantile=0.25,
            subspace_rank=1,
        )

        self.assertIsNotNone(fit)
        self.assertEqual(fit["fit_count"], 1)
        self.assertGreater(fit["direction"][0], 0.0)

    def test_choice_contrast_direction_tracks_scored_answer_category(self):
        trials = []
        vectors = {}
        for index, (category, value) in enumerate(
            [("private_target", 0.0), ("social", 2.0), ("other_clue_compatible", 1.0)]
        ):
            trial_id = f"choice_{index}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "pair_id": f"pair_{index}",
                    "case_id": f"case_{index}",
                    "variant": "ratio_memory",
                    "split": "train",
                    "m": 1,
                    "memory_total": 8,
                    "memory_ratio_label": "4:4",
                    "private_clue": "the number is prime",
                    "scored_choice_category": category,
                    "social_minus_private_number_logprob_margin": value,
                }
            )
            vectors[trial_id] = {1: np.array([value, 0.0])}

        fit = steering.fit_direction(
            trials=trials,
            vectors=vectors,
            layer=1,
            fit_split="train",
            direction_method="choice_contrast",
            direction_quantile=0.25,
            subspace_rank=1,
        )

        self.assertIsNotNone(fit)
        self.assertEqual(fit["choice_match_strategy"], "matched_m_ratio")
        self.assertGreater(fit["direction"][0], 0.0)

    def test_answer_category_vector_summary_saves_centroids_and_contrasts(self):
        trials = []
        vectors = {}
        for index, (category, value) in enumerate([("private_target", 0.0), ("social", 2.0)]):
            trial_id = f"category_{index}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "variant": "ratio_memory",
                    "split": "train",
                    "scored_choice_category": category,
                }
            )
            vectors[trial_id] = {1: np.array([value, 0.0])}

        rows, payload = steering.answer_category_vector_summary(
            trials=trials,
            vectors=vectors,
            layers=[1],
            fit_split="train",
        )

        self.assertIn("layer_1_centroid_social", payload)
        self.assertIn("layer_1_social_minus_private_target", payload)
        self.assertGreater(payload["layer_1_social_minus_private_target"][0], 0.0)
        self.assertTrue(any(row.get("kind") == "contrast" for row in rows))

    def test_ratio_social_counts_default_to_all_counts(self):
        self.assertEqual(steering.normalize_ratio_social_counts(4, []), [0, 1, 2, 3, 4])

    def test_generation_side_effect_summary_can_group_by_memory_ratio(self):
        rows = [
            {
                "dataset": "synthetic",
                "layer": 24,
                "calibrated_alpha": 0.0,
                "variant": "ratio_memory",
                "memory_ratio_label": ratio,
                "target_memory_count": ratio.split(":")[0],
                "social_memory_count": ratio.split(":")[1],
                "social_memory_fraction": social_fraction,
                "valid": True,
                "strict_json": True,
                "format_damage": False,
                "satisfies_private_clue": True,
                "choice_category": choice,
                "base_completion_perplexity": 1.0,
                "steered_completion_perplexity": 1.0,
            }
            for ratio, social_fraction, choice in [
                ("8:0", 0.0, "private_target"),
                ("0:8", 1.0, "social"),
            ]
        ]

        summary = steering.generation_side_effect_summary(
            rows,
            extra_group_fields=(
                "variant",
                "memory_ratio_label",
                "target_memory_count",
                "social_memory_count",
                "social_memory_fraction",
            ),
        )

        self.assertEqual(len(summary), 2)
        by_ratio = {row["memory_ratio_label"]: row for row in summary}
        self.assertEqual(by_ratio["8:0"]["private_target_choice_rate"], 1.0)
        self.assertEqual(by_ratio["0:8"]["social_choice_rate"], 1.0)

    def test_behavioral_summary_can_group_by_ratio(self):
        rows = [
            {
                "dataset": "synthetic",
                "layer": 24,
                "variant": "ratio_memory",
                "memory_ratio_label": "4:4",
                "calibrated_alpha": alpha,
                "valid_rate": 1.0,
                "format_damage_rate": 0.0,
                "satisfies_private_clue_rate": 0.8,
                "social_choice_rate": social_rate,
                "private_target_choice_rate": 0.5 - social_rate,
            }
            for alpha, social_rate in [(0.0, 0.1), (10.0, 0.3)]
        ]
        summary = steering.behavioral_steering_effect_summary(
            rows,
            extra_group_fields=("variant", "memory_ratio_label"),
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["memory_ratio_label"], "4:4")
        self.assertAlmostEqual(summary[0]["positive_social_choice_delta"], 0.2)

    def test_data_scaling_curve_uses_fit_object_counts(self):
        trials = []
        vectors = {}
        for case_index in range(2):
            for variant, value in [("target_memory", 0.0), ("social_memory", 1.0)]:
                trial_id = f"case_{case_index}_{variant}"
                trials.append(
                    {
                        "trial_id": trial_id,
                        "pair_id": f"case_{case_index}_m1_mem1",
                        "case_id": f"case_{case_index}",
                        "variant": variant,
                        "split": "train",
                        "social_minus_private_number_logprob_margin": value,
                    }
                )
                vectors[trial_id] = {1: np.array([value + case_index, value])}

        rows = steering.data_scaling_curve(
            trials=trials,
            vectors=vectors,
            layers=[1],
            fit_split="train",
            sizes=[1, 2],
            seed=0,
            direction_method="memory_contrast",
            direction_quantile=0.25,
            subspace_rank=1,
        )
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        self.assertTrue(ok_rows)
        self.assertTrue(all(row["fit_pair_count"] >= 1 for row in ok_rows))


if __name__ == "__main__":
    unittest.main()
