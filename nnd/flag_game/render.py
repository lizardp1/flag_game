from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from nnd.flag_game.catalog import COLOR_MAP, FlagSpec, ImageFlag, StripeFlag


def render_flag(flag: FlagSpec, width: int, height: int) -> np.ndarray:
    if isinstance(flag, ImageFlag):
        return _render_image_flag(flag, width=width, height=height)
    return _render_stripe_flag(flag, width=width, height=height)


def _render_image_flag(flag: ImageFlag, *, width: int, height: int) -> np.ndarray:
    path = Path(flag.image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing image-backed flag asset for {flag.country}: {path}. "
            "Populate assets with scripts/fetch_world_rectangle_flags.py."
        )

    with Image.open(path) as source:
        if source.mode in {"RGBA", "LA"} or (
            source.mode == "P" and "transparency" in source.info
        ):
            background = Image.new("RGBA", source.size, (255, 255, 255, 255))
            background.alpha_composite(source.convert("RGBA"))
            source = background.convert("RGB")
        else:
            source = source.convert("RGB")
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def _render_stripe_flag(flag: StripeFlag, *, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    stripe_count = len(flag.colors)

    if flag.orientation == "vertical":
        for idx, color_name in enumerate(flag.colors):
            start = (idx * width) // stripe_count
            end = ((idx + 1) * width) // stripe_count
            image[:, start:end, :] = COLOR_MAP[color_name]
    else:
        for idx, color_name in enumerate(flag.colors):
            start = (idx * height) // stripe_count
            end = ((idx + 1) * height) // stripe_count
            image[start:end, :, :] = COLOR_MAP[color_name]
    if flag.triangle_color is not None and flag.triangle_side == "left":
        tip_x = max(1, int(round(width * flag.triangle_width_fraction)))
        if height <= 1:
            max_x_by_row = np.full(height, tip_x, dtype=np.int32)
        else:
            y = np.arange(height, dtype=np.float32)
            centered = np.abs((2.0 * y / float(height - 1)) - 1.0)
            max_x_by_row = np.floor(tip_x * (1.0 - centered)).astype(np.int32)
        xs = np.arange(width, dtype=np.int32)[None, :]
        mask = xs <= max_x_by_row[:, None]
        image[mask] = COLOR_MAP[flag.triangle_color]
    return image


def image_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_uri(image: np.ndarray) -> str:
    payload = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path, format="PNG")
