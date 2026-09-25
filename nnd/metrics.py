from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out = []
    acc = 0.0
    queue = []
    for v in values:
        queue.append(v)
        acc += v
        if len(queue) > window:
            acc -= queue.pop(0)
        out.append(acc / len(queue))
    return out


def coordination_curve(matches: Iterable[bool], window: int) -> List[float]:
    values = [1.0 if m else 0.0 for m in matches]
    return moving_average(values, window)


def distribution_from_counts(counts: Dict[str, int], labels: List[str]) -> np.ndarray:
    total = sum(counts.values())
    if total <= 0:
        return np.ones(len(labels)) / len(labels)
    return np.array([counts.get(label, 0) for label in labels], dtype=float) / total


def polarization_u(p: np.ndarray) -> float:
    return float(np.sum(p * p))


def entropy(p: np.ndarray) -> float:
    p_safe = p + 1e-12
    return float(-np.sum(p_safe * np.log(p_safe)))
