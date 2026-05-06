from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np

from nnd.flag_game.catalog import COLOR_MAP, FlagSpec, StripeFlag
from nnd.flag_game.crops import all_crop_boxes, crop_image, scale_crop_box
from nnd.flag_game.render import render_flag


_INVERSE_COLOR_MAP = {tuple(rgb): name for name, rgb in COLOR_MAP.items()}


def build_crop_compatibility_cache(
    flags: Sequence[FlagSpec],
    *,
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    render_scale: int = 1,
) -> dict[str, set[bytes]]:
    scaled_boxes = [
        scale_crop_box(box, render_scale)
        for box in all_crop_boxes(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    ]
    render_width = canvas_width * render_scale
    render_height = canvas_height * render_scale
    cache: dict[str, set[bytes]] = {}
    for flag in flags:
        rendered = render_flag(flag, width=render_width, height=render_height)
        compatible_crops: set[bytes] = set()
        for box in scaled_boxes:
            compatible_crops.add(_crop_digest(crop_image(rendered, box)))
        cache[flag.country] = compatible_crops
    return cache


def infer_stripe_literal_signature(crop_image: np.ndarray) -> tuple[str, tuple[str, ...]]:
    if crop_image.size == 0:
        return "unclear", tuple()

    unique_colors = _visible_colors_in_scan_order(crop_image)
    if len(unique_colors) <= 1:
        return "unclear", unique_colors

    horizontal_change = float(np.mean(np.any(crop_image[:, 1:, :] != crop_image[:, :-1, :], axis=2)))
    vertical_change = float(np.mean(np.any(crop_image[1:, :, :] != crop_image[:-1, :, :], axis=2)))

    if horizontal_change > vertical_change * 1.5:
        return "vertical", _ordered_visible_colors(crop_image, orientation="vertical")
    if vertical_change > horizontal_change * 1.5:
        return "horizontal", _ordered_visible_colors(crop_image, orientation="horizontal")
    return "unclear", unique_colors


def compatible_countries_for_stripe_crop(
    crop_image: np.ndarray,
    *,
    flags: Sequence[StripeFlag],
    country_order: list[str] | None = None,
) -> list[str]:
    orientation, colors = infer_stripe_literal_signature(crop_image)
    if not colors:
        return []

    compatible: list[str] = []
    for flag in flags:
        if orientation == "unclear":
            if len(colors) == 1 and colors[0] in flag.colors:
                compatible.append(flag.country)
            continue
        if flag.orientation != orientation:
            continue
        if _is_contiguous_subsequence(flag.colors, colors):
            compatible.append(flag.country)

    if country_order is None:
        return compatible
    wanted = set(compatible)
    return [country for country in country_order if country in wanted]


def describe_crop_informativeness_fast(
    crop_image: np.ndarray,
    *,
    country_order: list[str],
    flags: Sequence[StripeFlag],
) -> dict[str, Any]:
    compatible_countries = compatible_countries_for_stripe_crop(
        crop_image,
        flags=flags,
        country_order=country_order,
    )
    return _describe_compatible_countries(
        compatible_countries,
        pool_size=len(country_order),
    )


def describe_crop_informativeness(
    crop_image: np.ndarray,
    *,
    country_order: list[str],
    compatibility_cache: dict[str, set[bytes]],
) -> dict[str, Any]:
    crop_key = _crop_digest(crop_image)
    compatible_countries = [
        country
        for country in country_order
        if crop_key in compatibility_cache.get(country, set())
    ]
    return _describe_compatible_countries(
        compatible_countries,
        pool_size=len(country_order),
    )


def _describe_compatible_countries(
    compatible_countries: list[str],
    *,
    pool_size: int,
) -> dict[str, Any]:
    compatible_count = len(compatible_countries)
    if compatible_count <= 0 or pool_size <= 0:
        return {
            "pool_size": pool_size,
            "compatible_country_count": compatible_count,
            "compatible_countries": compatible_countries,
            "ambiguity_fraction": None,
            "informativeness_bits": None,
            "informativeness_score": None,
            "informativeness_label": "invalid",
            "is_unique": False,
        }

    ambiguity_fraction = compatible_count / float(pool_size)
    informativeness_bits = math.log2(pool_size / compatible_count)
    if pool_size == 1:
        informativeness_score = 1.0
    else:
        informativeness_score = 1.0 - (math.log(compatible_count) / math.log(pool_size))

    return {
        "pool_size": pool_size,
        "compatible_country_count": compatible_count,
        "compatible_countries": compatible_countries,
        "ambiguity_fraction": ambiguity_fraction,
        "informativeness_bits": informativeness_bits,
        "informativeness_score": informativeness_score,
        "informativeness_label": classify_crop_informativeness(compatible_count, pool_size),
        "is_unique": compatible_count == 1,
    }


def _crop_digest(crop_image: np.ndarray) -> bytes:
    return hashlib.blake2b(crop_image.tobytes(), digest_size=16).digest()


def _visible_colors_in_scan_order(crop_image: np.ndarray) -> tuple[str, ...]:
    colors: list[str] = []
    seen: set[str] = set()
    for pixel in crop_image.reshape(-1, 3):
        name = _INVERSE_COLOR_MAP.get(tuple(int(value) for value in pixel))
        if name is None or name in seen:
            continue
        colors.append(name)
        seen.add(name)
    return tuple(colors)


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


def _is_contiguous_subsequence(full: Sequence[str], subseq: Sequence[str]) -> bool:
    if not subseq or len(subseq) > len(full):
        return False
    size = len(subseq)
    for start in range(0, len(full) - size + 1):
        if tuple(full[start : start + size]) == tuple(subseq):
            return True
    return False


def classify_crop_informativeness(compatible_count: int, pool_size: int) -> str:
    if compatible_count <= 0 or pool_size <= 0:
        return "invalid"
    if compatible_count == 1:
        return "unique"
    if compatible_count <= 3 or compatible_count / float(pool_size) <= 0.15:
        return "narrow"
    if compatible_count <= 6 or compatible_count / float(pool_size) <= 0.35:
        return "moderate"
    return "ambiguous"
