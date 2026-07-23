from __future__ import annotations

import random
import unittest

from nnd.number_game import prompts
from nnd.number_game.conflict import _memory_lines
from nnd.number_game.domain import candidate_numbers, clue_information_bits, clue_information_phase
from nnd.number_game.parsing import NumberMessage, ParseError, parse_number_message
from nnd.number_game.runner import _stable_probe_consensus


class NumberGameMemoryFormatTest(unittest.TestCase):
    def test_pairwise_prompt_memory_entries_are_bare_lines(self) -> None:
        text = prompts.interaction_text(
            numbers=[1, 2, 3],
            private_clue="the number is odd",
            memory_lines=["12 | The number is even.", "7 | I am confident the answer is 7."],
            m=3,
            prompt_social_susceptibility=False,
        )

        self.assertIn(
            "Transcript memory (oldest -> newest):\n"
            "- 12 | The number is even.\n"
            "- 7 | I am confident the answer is 7.",
            text,
        )
        self.assertNotIn("A02 said", text)
        self.assertNotIn("agent 2", text)
        self.assertIn('Output JSON exactly: {"number":<integer>,"reason":"<one sentence>"}', text)
        self.assertNotIn('"hint"', text)
        self.assertNotIn("sampled uniformly", text)

    def test_m1_and_m3_memory_formats_are_not_mixed(self) -> None:
        prompts.interaction_text(
            numbers=[1, 2, 3],
            private_clue="the number is odd",
            memory_lines=["12", "7"],
            m=1,
            prompt_social_susceptibility=False,
        )
        prompts.interaction_text(
            numbers=[1, 2, 3],
            private_clue="the number is odd",
            memory_lines=["12 | The number is even.", "7 | The number is prime."],
            m=3,
            prompt_social_susceptibility=False,
        )
        with self.assertRaises(ValueError):
            prompts.interaction_text(
                numbers=[1, 2, 3],
                private_clue="the number is odd",
                memory_lines=["12 | The number is even."],
                m=1,
                prompt_social_susceptibility=False,
            )
        with self.assertRaises(ValueError):
            prompts.interaction_text(
                numbers=[1, 2, 3],
                private_clue="the number is odd",
                memory_lines=["12"],
                m=3,
                prompt_social_susceptibility=False,
            )

    def test_prompt_can_include_integer_range_without_allowed_list(self) -> None:
        text = prompts.interaction_text(
            numbers=list(range(1, 101)),
            private_clue="the number is odd",
            memory_lines=[],
            m=1,
            prompt_social_susceptibility=False,
            prompt_number_range=True,
        )

        self.assertIn("The hidden integer is in the range 1 through 100, inclusive.", text)
        self.assertNotIn("sampled", text)
        self.assertNotIn("uniformly", text)
        self.assertNotIn("Allowed numbers", text)
        self.assertNotIn("[1, 2, 3", text)

    def test_normalized_memory_entry_matches_flag_style(self) -> None:
        self.assertEqual(NumberMessage(number=12).normalized_memory_entry(), "12")
        self.assertEqual(
            NumberMessage(number=12, reason="My clue points to 12.").normalized_memory_entry(),
            "12 | My clue points to 12.",
        )

    def test_m3_parser_uses_reason_field(self) -> None:
        parsed = parse_number_message(
            '{"number":12,"reason":"My clue points to 12."}',
            allowed_numbers=[],
            m=3,
        )
        self.assertEqual(parsed.number, 12)
        self.assertEqual(parsed.reason, "My clue points to 12.")

        with self.assertRaises(ParseError):
            parse_number_message(
                '{"number":12,"hint":"My clue points to 12."}',
                allowed_numbers=[],
                m=3,
            )

    def test_conflict_probe_memory_entries_are_bare_lines(self) -> None:
        lines = _memory_lines(
            truth_number=7,
            social_number=12,
            m=3,
            target_count=1,
            social_count=2,
            rng=random.Random(0),
        )

        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.startswith(("7", "12")) for line in lines))
        self.assertTrue(all(" | " in line for line in lines))
        self.assertTrue(all("agent" not in line.lower() for line in lines))

    def test_pairwise_early_stop_uses_full_agent_probe_consensus(self) -> None:
        rows = []
        for t, number in [(0, 7), (4, 7), (8, 7), (12, 7), (16, 7)]:
            for agent_id in range(4):
                rows.append({"t": t, "agent_id": agent_id, "valid": True, "number": number})

        self.assertEqual(
            _stable_probe_consensus(rows, n_agents=4, window=5, consensus_threshold=0.9),
            (True, 7),
        )

        rows[-1]["number"] = 12
        self.assertEqual(
            _stable_probe_consensus(rows, n_agents=4, window=5, consensus_threshold=0.9),
            (False, None),
        )

    def test_clue_information_bits_track_candidate_set_size(self) -> None:
        numbers = candidate_numbers(1, 30)

        self.assertAlmostEqual(clue_information_bits(numbers, "the number is odd"), 1.0)
        self.assertEqual(clue_information_phase(clue_information_bits(numbers, "the number is odd")), "weak")
        self.assertEqual(clue_information_phase(clue_information_bits(numbers, "the number is prime")), "medium")
        self.assertEqual(
            clue_information_phase(clue_information_bits(numbers, "the digits sum to 7")),
            "strong",
        )


if __name__ == "__main__":
    unittest.main()
