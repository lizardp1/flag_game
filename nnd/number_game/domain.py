from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable


@dataclass(frozen=True)
class NumberClue:
    name: str
    text: str
    predicate: Callable[[int], bool]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(math.sqrt(value)) + 1):
        if value % divisor == 0:
            return False
    return True


def _digits_sum_to(total: int) -> Callable[[int], bool]:
    return lambda value: sum(int(char) for char in str(abs(value))) == total


DEFAULT_CLUES: tuple[NumberClue, ...] = (
    NumberClue("prime", "the number is prime", _is_prime),
    NumberClue("odd", "the number is odd", lambda value: value % 2 == 1),
    NumberClue("even", "the number is even", lambda value: value % 2 == 0),
    NumberClue("under_20", "the number is under 20", lambda value: value < 20),
    NumberClue("over_10", "the number is over 10", lambda value: value > 10),
    NumberClue("multiple_of_3", "the number is a multiple of 3", lambda value: value % 3 == 0),
    NumberClue("multiple_of_5", "the number is a multiple of 5", lambda value: value % 5 == 0),
    NumberClue("one_digit", "the number has one digit", lambda value: 0 <= value <= 9),
    NumberClue("two_digits", "the number has two digits", lambda value: 10 <= value <= 99),
    NumberClue("square", "the number is a perfect square", lambda value: int(math.sqrt(value)) ** 2 == value),
    NumberClue("digit_sum_7", "the digits sum to 7", _digits_sum_to(7)),
    NumberClue("digit_sum_10", "the digits sum to 10", _digits_sum_to(10)),
)


def candidate_numbers(min_number: int, max_number: int) -> list[int]:
    return list(range(min_number, max_number + 1))


def matching_clues(number: int, clue_pool: tuple[NumberClue, ...] = DEFAULT_CLUES) -> list[NumberClue]:
    return [clue for clue in clue_pool if clue.predicate(number)]


def sample_truth_and_clues(
    *,
    rng: random.Random,
    n_agents: int,
    min_number: int,
    max_number: int,
    fixed_truth_number: int | None = None,
) -> tuple[int, list[NumberClue]]:
    numbers = candidate_numbers(min_number, max_number)
    if fixed_truth_number is not None:
        if fixed_truth_number not in numbers:
            raise ValueError("fixed_truth_number must be inside the configured number range")
        truth = fixed_truth_number
    else:
        viable = [number for number in numbers if len(matching_clues(number)) >= min(n_agents, 2)]
        truth = rng.choice(viable or numbers)

    truth_clues = matching_clues(truth)
    if not truth_clues:
        truth_clues = [NumberClue("range_only", f"the number is between {min_number} and {max_number}", lambda _: True)]
    return truth, [rng.choice(truth_clues) for _ in range(n_agents)]


def filter_candidates(numbers: list[int], clue_texts: list[str]) -> list[int]:
    by_text = {clue.text: clue for clue in DEFAULT_CLUES}
    candidates = list(numbers)
    for text in clue_texts:
        clue = by_text.get(text)
        if clue is not None:
            candidates = [number for number in candidates if clue.predicate(number)]
    return candidates


def clue_candidates(numbers: list[int], clue_text: str) -> list[int]:
    return filter_candidates(numbers, [clue_text])


def clue_information_bits(numbers: list[int], clue_text: str) -> float | None:
    if not numbers:
        return None
    candidates = clue_candidates(numbers, clue_text)
    if not candidates:
        return None
    return math.log2(len(numbers) / float(len(candidates)))


def clue_information_phase(bits: float | None) -> str:
    if bits is None:
        return "unknown"
    if bits <= 1.0:
        return "weak"
    if bits <= 2.0:
        return "medium"
    return "strong"


def clue_matches_number(clue_text: str, number: int) -> bool | None:
    clue = {clue.text: clue for clue in DEFAULT_CLUES}.get(clue_text)
    if clue is None:
        return None
    return clue.predicate(number)
