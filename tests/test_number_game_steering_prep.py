import unittest

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


if __name__ == "__main__":
    unittest.main()
