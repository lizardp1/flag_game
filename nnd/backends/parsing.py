from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from nnd.prompts import PAD_TOKEN


class ParseError(ValueError):
    pass


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _fuzzy_match(val: str, labels: List[str], name: str = "label") -> str:
    """Exact match first, then closest label if edit distance <= 2."""
    if val in labels:
        return val
    best_label, best_dist = None, float("inf")
    for lab in labels:
        d = _edit_distance(val, lab)
        if d < best_dist:
            best_label, best_dist = lab, d
    if best_dist <= 2:
        return best_label  # type: ignore[return-value]
    raise ParseError(
        f"{name} value '{val}' not in allowed labels (closest: '{best_label}', dist={best_dist})"
    )


def _load_json_strict(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ParseError("Response is not a string")
    stripped = text.strip()
    if not stripped:
        raise ParseError("Empty response")
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(stripped)
        if extracted is None:
            raise ParseError("Response is not valid JSON") from exc
        try:
            obj = json.loads(extracted)
        except json.JSONDecodeError as exc2:
            raise ParseError("Response is not valid JSON") from exc2
    if not isinstance(obj, dict):
        raise ParseError("JSON response must be an object")
    return obj


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()


def parse_label_response(text: str, labels: List[str]) -> str:
    obj = _load_json_strict(text)
    if "label" not in obj:
        raise ParseError("Missing 'label' key")
    if len(obj) != 1:
        raise ParseError("Unexpected keys in response")
    label = obj["label"]
    if not isinstance(label, str):
        raise ParseError("Label must be a string")
    if label == PAD_TOKEN:
        raise ParseError("Label must not be the pad token")
    label = _fuzzy_match(label, labels)
    return label


def parse_raw_label_response(text: str, labels: List[str]) -> str:
    label = text.strip()
    if not label:
        raise ParseError("Missing label")
    if label == PAD_TOKEN:
        raise ParseError("Label must not be the pad token")
    label = _fuzzy_match(label, labels)
    return label


def parse_topm_response(text: str, labels: List[str], m: int) -> List[str]:
    obj = _load_json_strict(text)
    if "labels" not in obj:
        raise ParseError("Missing 'labels' key")
    if len(obj) != 1:
        raise ParseError("Unexpected keys in response")
    values = obj["labels"]
    if not isinstance(values, list):
        raise ParseError("'labels' must be a list")
    if len(values) != m:
        raise ParseError(f"'labels' must have length {m}")
    for i, label in enumerate(values):
        if not isinstance(label, str):
            raise ParseError("Each label must be a string")
        if label == PAD_TOKEN:
            raise ParseError("Label must not be the pad token")
        values[i] = _fuzzy_match(label, labels)
    return values
