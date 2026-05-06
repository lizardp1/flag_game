from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nnd.flag_game.catalog import COLOR_MAP, StripeFlag, get_flag
from nnd.flag_game.crops import CropBox, all_crop_boxes, crop_image
from nnd.flag_game.diagnostics import classify_crop_informativeness
from nnd.flag_game.render import render_flag


@dataclass(frozen=True)
class OrientationProbeTemplate:
    name: str
    label: str
    truth_country: str
    expected_orientation: str
    expected_visible_color_order: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    label: str
    tile_width: int
    tile_height: int


DEFAULT_ORIENTATION_PROBE_TEMPLATES: tuple[OrientationProbeTemplate, ...] = (
    OrientationProbeTemplate(
        name="austria_horizontal_unique",
        label="Austria Horizontal Unique",
        truth_country="Austria",
        expected_orientation="horizontal",
        expected_visible_color_order=("white", "red"),
        note="Unique horizontal white-red boundary crop.",
    ),
    OrientationProbeTemplate(
        name="france_vertical_narrow",
        label="France Vertical Narrow",
        truth_country="France",
        expected_orientation="vertical",
        expected_visible_color_order=("white", "red"),
        note="Narrow vertical white-red boundary crop with multiple compatible countries.",
    ),
    OrientationProbeTemplate(
        name="chad_vertical_unique",
        label="Chad Vertical Unique",
        truth_country="Chad",
        expected_orientation="vertical",
        expected_visible_color_order=("blue", "yellow"),
        note="Unique vertical blue-yellow boundary crop.",
    ),
    OrientationProbeTemplate(
        name="ireland_single_color",
        label="Ireland Single Color",
        truth_country="Ireland",
        expected_orientation="unclear",
        expected_visible_color_order=("orange",),
        note="Single-color orange crop that should remain orientation-agnostic.",
    ),
)


DEFAULT_SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec(name="2x8", label="2x8 Tall", tile_width=2, tile_height=8),
    ShapeSpec(name="3x6", label="3x6 Tall", tile_width=3, tile_height=6),
    ShapeSpec(name="3x3", label="3x3 Square", tile_width=3, tile_height=3),
    ShapeSpec(name="6x3", label="6x3 Wide", tile_width=6, tile_height=3),
)


_INVERSE_COLOR_MAP = {tuple(rgb): name for name, rgb in COLOR_MAP.items()}


def parse_shape_spec(text: str) -> ShapeSpec:
    match = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if match is None:
        raise ValueError(f"Invalid shape {text!r}; expected WIDTHxHEIGHT such as 3x6")
    tile_width = int(match.group(1))
    tile_height = int(match.group(2))
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Shape dimensions must be positive")
    return ShapeSpec(
        name=f"{tile_width}x{tile_height}",
        label=f"{tile_width}x{tile_height}",
        tile_width=tile_width,
        tile_height=tile_height,
    )


def visible_colors_in_scan_order(crop_image: np.ndarray) -> tuple[str, ...]:
    colors: list[str] = []
    seen: set[str] = set()
    for pixel in crop_image.reshape(-1, 3):
        name = _INVERSE_COLOR_MAP.get(tuple(int(value) for value in pixel))
        if name is None or name in seen:
            continue
        colors.append(name)
        seen.add(name)
    return tuple(colors)


def infer_literal_signature(crop_image: np.ndarray) -> tuple[str, tuple[str, ...]]:
    if crop_image.size == 0:
        return "unclear", tuple()

    unique_colors = visible_colors_in_scan_order(crop_image)
    if len(unique_colors) <= 1:
        return "unclear", unique_colors

    horizontal_change = float(np.mean(np.any(crop_image[:, 1:, :] != crop_image[:, :-1, :], axis=2)))
    vertical_change = float(np.mean(np.any(crop_image[1:, :, :] != crop_image[:-1, :, :], axis=2)))

    if horizontal_change > vertical_change * 1.5:
        return "vertical", _ordered_visible_colors(crop_image, orientation="vertical")
    if vertical_change > horizontal_change * 1.5:
        return "horizontal", _ordered_visible_colors(crop_image, orientation="horizontal")
    return "unclear", unique_colors


def _ordered_visible_colors(crop_image: np.ndarray, *, orientation: str) -> tuple[str, ...]:
    colors: list[str] = []
    if orientation == "horizontal":
        for row in range(crop_image.shape[0]):
            rgb = tuple(int(value) for value in crop_image[row, 0, :])
            name = _INVERSE_COLOR_MAP.get(rgb)
            if name is not None and (not colors or colors[-1] != name):
                colors.append(name)
        return tuple(colors)
    if orientation == "vertical":
        for col in range(crop_image.shape[1]):
            rgb = tuple(int(value) for value in crop_image[0, col, :])
            name = _INVERSE_COLOR_MAP.get(rgb)
            if name is not None and (not colors or colors[-1] != name):
                colors.append(name)
        return tuple(colors)
    raise ValueError(f"Unsupported orientation: {orientation}")


def choose_logical_probe_box(
    *,
    template: OrientationProbeTemplate,
    tile_width: int,
    tile_height: int,
    countries: Sequence[StripeFlag],
    canvas_width: int,
    canvas_height: int,
    render_scale: int = 1,
) -> tuple[CropBox, dict[str, object]]:
    truth_flag = get_flag(template.truth_country)
    if render_scale < 1:
        raise ValueError("render_scale must be >= 1")
    full_image = render_flag(
        truth_flag,
        width=canvas_width * render_scale,
        height=canvas_height * render_scale,
    )

    candidates: list[tuple[tuple[float, int, int, int], CropBox, dict[str, object]]] = []
    canvas_center_y = (canvas_height - 1) / 2.0
    canvas_center_x = (canvas_width - 1) / 2.0
    for box in all_crop_boxes(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tile_width=tile_width,
        tile_height=tile_height,
    ):
        scaled_box = CropBox(
            crop_index=box.crop_index,
            top=box.top * render_scale,
            left=box.left * render_scale,
            height=box.height * render_scale,
            width=box.width * render_scale,
        )
        crop = crop_image(full_image, scaled_box)
        orientation, colors = infer_literal_signature(crop)
        if orientation != template.expected_orientation:
            continue
        if tuple(colors) != tuple(template.expected_visible_color_order):
            continue
        compatible_countries = compatible_countries_for_literal_signature(
            countries=countries,
            orientation=orientation,
            colors=colors,
        )
        if template.truth_country not in compatible_countries:
            continue
        compatible_count = len(compatible_countries)
        pool_size = len(countries)
        if compatible_count <= 0:
            continue
        informativeness_bits = math.log2(pool_size / compatible_count)
        if pool_size == 1:
            informativeness_score = 1.0
        else:
            informativeness_score = 1.0 - (math.log(compatible_count) / math.log(pool_size))
        diagnostic = {
            "pool_size": pool_size,
            "compatible_country_count": compatible_count,
            "compatible_countries": compatible_countries,
            "ambiguity_fraction": compatible_count / float(pool_size),
            "informativeness_bits": informativeness_bits,
            "informativeness_score": informativeness_score,
            "informativeness_label": classify_crop_informativeness(compatible_count, pool_size),
            "is_unique": compatible_count == 1,
        }
        center_y = box.top + (box.height - 1) / 2.0
        center_x = box.left + (box.width - 1) / 2.0
        center_distance = math.hypot(center_y - canvas_center_y, center_x - canvas_center_x)
        key = (
            center_distance,
            int(diagnostic["compatible_country_count"]),
            int(box.top),
            int(box.left),
        )
        candidates.append((key, box, diagnostic))

    if not candidates:
        raise ValueError(
            "No matching logical crop found for "
            f"{template.name} with tile {tile_width}x{tile_height}"
        )

    candidates.sort(key=lambda item: item[0])
    _, box, diagnostic = candidates[0]
    return box, diagnostic


def compatible_countries_for_literal_signature(
    *,
    countries: Sequence[StripeFlag],
    orientation: str,
    colors: Sequence[str],
) -> list[str]:
    if not colors:
        return []
    color_tuple = tuple(colors)
    compatible: list[str] = []
    for flag in countries:
        if orientation == "unclear":
            if len(color_tuple) == 1 and color_tuple[0] in flag.colors:
                compatible.append(flag.country)
            continue
        if flag.orientation != orientation:
            continue
        if _is_contiguous_subsequence(flag.colors, color_tuple):
            compatible.append(flag.country)
    return compatible


def _is_contiguous_subsequence(full: Sequence[str], subseq: Sequence[str]) -> bool:
    if not subseq or len(subseq) > len(full):
        return False
    size = len(subseq)
    for start in range(0, len(full) - size + 1):
        if tuple(full[start : start + size]) == tuple(subseq):
            return True
    return False


def render_scaled_crop(
    *,
    truth_country: str,
    box: CropBox,
    render_scale: int,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    if render_scale < 1:
        raise ValueError("render_scale must be >= 1")
    full_image = render_flag(
        get_flag(truth_country),
        width=canvas_width * render_scale,
        height=canvas_height * render_scale,
    )
    scaled_box = CropBox(
        crop_index=box.crop_index,
        top=box.top * render_scale,
        left=box.left * render_scale,
        height=box.height * render_scale,
        width=box.width * render_scale,
    )
    return crop_image(full_image, scaled_box)


def normalize_country_guess(raw_country: object, countries: Iterable[str]) -> str | None:
    if not isinstance(raw_country, str):
        return None
    stripped = raw_country.strip()
    if not stripped:
        return None
    country_list = list(countries)
    if stripped in country_list:
        return stripped

    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    normalized = _normalize(stripped)
    matches = [country for country in country_list if _normalize(country) == normalized]
    if len(matches) == 1:
        return matches[0]
    return None


def first_passing_scales(
    summary_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_col: str,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    for group_key, group in summary_df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key)}
        ordered = group.sort_values("render_scale")
        passing = ordered[ordered[metric_col] >= threshold]
        row[f"{metric_col}_first_passing_scale"] = (
            int(passing.iloc[0]["render_scale"]) if not passing.empty else None
        )
        rows.append(row)
    return pd.DataFrame(rows)
