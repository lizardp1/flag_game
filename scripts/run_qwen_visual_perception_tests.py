#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from nnd.flag_game.catalog import COLOR_MAP
from nnd.flag_game.render import save_png


DEFAULT_MODEL_IDS = ["Qwen/Qwen2.5-VL-7B-Instruct"]
FLAG_CORE_COLORS = ["red", "blue", "green", "white", "black", "yellow", "orange", "light_blue"]
FLAG_EXTENDED_COLORS = [
    "red",
    "blue",
    "green",
    "white",
    "black",
    "yellow",
    "orange",
    "light_blue",
    "navy",
    "gold",
    "cyan",
    "teal",
]
LEGACY_COLORS = ["red", "blue", "green", "white", "black", "yellow", "orange", "purple"]
COLOR_SETS = {
    "flag_core": FLAG_CORE_COLORS,
    "flag_extended": FLAG_EXTENDED_COLORS,
    "legacy": LEGACY_COLORS,
}
DEFAULT_COLOR_SET = "flag_core"
DEFAULT_COLORS = COLOR_SETS[DEFAULT_COLOR_SET]
DEFAULT_PIXEL_SIZES = ["24x16", "48x32", "75x150", "150x100", "300x200", "600x400"]
DEFAULT_STRIPE_PATTERNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "vertical_france": ("vertical", ("blue", "white", "red")),
    "vertical_ireland": ("vertical", ("green", "white", "orange")),
    "vertical_belgium": ("vertical", ("black", "yellow", "red")),
    "vertical_nigeria": ("vertical", ("green", "white", "green")),
    "horizontal_austria": ("horizontal", ("red", "white", "red")),
    "horizontal_russia": ("horizontal", ("white", "blue", "red")),
    "horizontal_ukraine": ("horizontal", ("blue", "yellow")),
}

COLOR_ALIASES = {
    "grey": "gray",
    "lightblue": "light_blue",
    "light_blue": "light_blue",
    "light-blue": "light_blue",
    "light blue": "light_blue",
    "darkblue": "blue",
    "dark_blue": "blue",
    "dark-blue": "blue",
    "dark blue": "blue",
    "navyblue": "navy",
    "navy_blue": "navy",
    "navy-blue": "navy",
    "navy blue": "navy",
    "golden": "gold",
    "golden yellow": "yellow",
}

WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    task_type: str
    width: int
    height: int
    image: np.ndarray
    expected: dict[str, Any]
    prompt: str
    artifact_relpath: str

    def metadata(self) -> dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "task_type": self.task_type,
            "width": self.width,
            "height": self.height,
            "area_pixels": self.width * self.height,
            "expected": self.expected,
            "prompt": self.prompt,
            "artifact_relpath": self.artifact_relpath,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run visual-only perception tests for local open VLMs: solid colors, "
            "stripe orientation/count/order, and pixel-size sweeps."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "qwen", "qwen2_5", "qwen3", "llava", "kimi_vl", "oracle"],
        default="auto",
        help=(
            "Use auto to infer a loader from the model id. Use qwen, qwen2_5, "
            "qwen3, llava, or kimi_vl to force a loader, or oracle for local "
            "output/summary smoke tests."
        ),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Hugging Face model id. Repeat or comma-separate. Defaults to Qwen2.5-VL-7B.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "qwen_visual_perception")
    parser.add_argument("--suite", choices=["all", "colors", "stripes"], default="all")
    parser.add_argument(
        "--color-set",
        choices=sorted(COLOR_SETS),
        default=DEFAULT_COLOR_SET,
        help=(
            "Named solid-color preset used when --colors is omitted. flag_core is "
            "the recommended flag-color battery."
        ),
    )
    parser.add_argument(
        "--colors",
        action="append",
        default=None,
        help=(
            "Override solid-color tests. Repeat or comma-separate. Must be keys in "
            "COLOR_MAP."
        ),
    )
    parser.add_argument(
        "--pixel-sizes",
        action="append",
        default=None,
        help="Image sizes as WIDTHxHEIGHT. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--stripe-patterns",
        action="append",
        default=None,
        help=(
            "Stripe pattern names or specs. Names include vertical_france, horizontal_ukraine. "
            "Specs use orientation:color-color-color, e.g. vertical:blue-white-red."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--break-threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold used in breakpoints.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render stimuli and prompts, then exit before loading any model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_tests is not None and args.max_tests < 1:
        raise ValueError("--max-tests must be >= 1")

    model_ids = split_values(args.model_id, DEFAULT_MODEL_IDS)
    colors = parse_colors(resolve_color_values(args))
    sizes = [parse_pixel_size(value) for value in split_values(args.pixel_sizes, DEFAULT_PIXEL_SIZES)]
    stripe_patterns = parse_stripe_patterns(args.stripe_patterns)

    out_dir = args.out
    artifact_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stimuli = build_stimuli(
        suite=args.suite,
        colors=colors,
        sizes=sizes,
        stripe_patterns=stripe_patterns,
    )
    if args.max_tests is not None:
        stimuli = stimuli[: args.max_tests]
    save_stimuli(artifact_dir, out_dir, stimuli)

    if args.dry_run:
        write_json(
            out_dir / "summary.json",
            {
                "status": "dry_run",
                "stimulus_count": len(stimuli),
                "model_ids": model_ids,
                "outputs": ["stimuli.jsonl"],
            },
        )
        print(f"Dry run complete. Rendered {len(stimuli)} stimuli to {out_dir}")
        return

    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        print(f"Starting visual perception tests for {model_id}")
        runner = build_runner(args, model_id)
        for stimulus in stimuli:
            image_path = (artifact_dir / stimulus.artifact_relpath).resolve()
            for repeat in range(args.repeats):
                raw_output = runner(stimulus, image_path)
                row = build_result_row(
                    model_id=model_id,
                    backend=args.backend,
                    stimulus=stimulus,
                    image_path=image_path,
                    repeat=repeat,
                    raw_output=raw_output,
                )
                rows.append(row)
                print(format_progress(row))
        release_runner(runner)

    write_outputs(out_dir, rows, args.break_threshold)
    print(f"Wrote visual perception results to {out_dir}")


def split_values(values: list[str] | None, default: list[str]) -> list[str]:
    if values is None:
        return list(default)
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(","))
    return [item for item in items if item.strip()]


def resolve_color_values(args: argparse.Namespace) -> list[str]:
    if args.colors is not None:
        return split_values(args.colors, [])
    return list(COLOR_SETS[args.color_set])


def parse_colors(values: list[str]) -> list[str]:
    colors = [normalize_color(value) for value in values]
    unknown = sorted({color for color in colors if color not in COLOR_MAP})
    if unknown:
        valid = ", ".join(sorted(COLOR_MAP))
        raise ValueError(f"Unknown colors {unknown}; valid colors are: {valid}")
    return colors


def parse_pixel_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value.lower())
    if match is None:
        raise ValueError(f"Invalid pixel size {value!r}; use WIDTHxHEIGHT, e.g. 75x150")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1:
        raise ValueError(f"Invalid pixel size {value!r}; dimensions must be positive")
    return width, height


def parse_stripe_patterns(values: list[str] | None) -> dict[str, tuple[str, tuple[str, ...]]]:
    if values is None:
        return dict(DEFAULT_STRIPE_PATTERNS)
    patterns: dict[str, tuple[str, tuple[str, ...]]] = {}
    for raw in split_values(values, []):
        if raw in DEFAULT_STRIPE_PATTERNS:
            patterns[raw] = DEFAULT_STRIPE_PATTERNS[raw]
            continue
        match = re.fullmatch(r"(vertical|horizontal):([A-Za-z0-9_\-\s]+)", raw.strip())
        if match is None:
            valid = ", ".join(sorted(DEFAULT_STRIPE_PATTERNS))
            raise ValueError(
                f"Unknown stripe pattern {raw!r}. Use a known pattern ({valid}) "
                "or a spec like vertical:blue-white-red."
            )
        orientation = match.group(1)
        colors = tuple(normalize_color(part) for part in match.group(2).split("-"))
        unknown = sorted({color for color in colors if color not in COLOR_MAP})
        if unknown:
            raise ValueError(f"Unknown colors in {raw!r}: {unknown}")
        if len(colors) < 2:
            raise ValueError(f"Stripe pattern {raw!r} must contain at least two colors")
        name = f"{orientation}_{'_'.join(colors)}"
        patterns[name] = (orientation, colors)
    return patterns


def build_stimuli(
    *,
    suite: str,
    colors: list[str],
    sizes: list[tuple[int, int]],
    stripe_patterns: dict[str, tuple[str, tuple[str, ...]]],
) -> list[Stimulus]:
    stimuli: list[Stimulus] = []
    if suite in {"all", "colors"}:
        for width, height in sizes:
            for color in colors:
                stimulus_id = f"color__{color}__{width}x{height}"
                stimuli.append(
                    Stimulus(
                        stimulus_id=stimulus_id,
                        task_type="color",
                        width=width,
                        height=height,
                        image=render_solid_color(color, width=width, height=height),
                        expected={"color": color, "color_group": color_group(color)},
                        prompt=color_prompt(),
                        artifact_relpath=f"colors/{stimulus_id}.png",
                    )
                )
    if suite in {"all", "stripes"}:
        for width, height in sizes:
            for pattern_name, (orientation, pattern_colors) in stripe_patterns.items():
                stimulus_id = f"stripes__{pattern_name}__{width}x{height}"
                stimuli.append(
                    Stimulus(
                        stimulus_id=stimulus_id,
                        task_type="stripes",
                        width=width,
                        height=height,
                        image=render_stripes(
                            orientation=orientation,
                            colors=pattern_colors,
                            width=width,
                            height=height,
                        ),
                        expected={
                            "pattern_name": pattern_name,
                            "orientation": orientation,
                            "stripe_count": len(pattern_colors),
                            "colors": list(pattern_colors),
                        },
                        prompt=stripe_prompt(),
                        artifact_relpath=f"stripes/{stimulus_id}.png",
                    )
                )
    return stimuli


def render_solid_color(color: str, *, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = COLOR_MAP[color]
    return image


def color_group(color: str) -> str:
    if color in {"red", "blue", "yellow"}:
        return "primary_flag_color"
    if color in {"black", "white"}:
        return "neutral_flag_color"
    if color in {"green", "orange", "light_blue"}:
        return "non_primary_core_flag_color"
    if color in {"navy", "gold", "cyan", "teal"}:
        return "extended_flag_color"
    return "other_catalog_color"


def render_stripes(
    *,
    orientation: str,
    colors: tuple[str, ...],
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    stripe_count = len(colors)
    if orientation == "vertical":
        for idx, color in enumerate(colors):
            start = (idx * width) // stripe_count
            end = ((idx + 1) * width) // stripe_count
            image[:, start:end, :] = COLOR_MAP[color]
        return image
    if orientation == "horizontal":
        for idx, color in enumerate(colors):
            start = (idx * height) // stripe_count
            end = ((idx + 1) * height) // stripe_count
            image[start:end, :, :] = COLOR_MAP[color]
        return image
    raise ValueError(f"Unsupported orientation: {orientation}")


def color_prompt() -> str:
    return (
        "This is a synthetic visual perception test, not a flag-identification task.\n"
        "The image is intended to be one solid color rectangle.\n"
        f"Allowed color names: {', '.join(sorted(COLOR_MAP))}.\n"
        'Return JSON exactly: {"color":"<one allowed color>"}'
    )


def stripe_prompt() -> str:
    return (
        "This is a synthetic visual perception test, not a flag-identification task.\n"
        "The image is intended to be a simple rectangle made of uniform color stripes.\n"
        "Use vertical when stripes are arranged left-to-right; use horizontal when "
        "stripes are arranged top-to-bottom.\n"
        f"Allowed color names: {', '.join(sorted(COLOR_MAP))}.\n"
        'Return JSON exactly: {"orientation":"vertical|horizontal",'
        '"stripe_count":<integer>,"colors":["<colors in visual order>"]}'
    )


def system_prompt() -> str:
    return (
        "You must output only valid JSON. No markdown, no prose outside the JSON object, "
        "and no country names. Answer only the low-level visual question."
    )


def save_stimuli(artifact_dir: Path, out_dir: Path, stimuli: list[Stimulus]) -> None:
    with open(out_dir / "stimuli.jsonl", "w") as handle:
        for stimulus in stimuli:
            save_png(artifact_dir / stimulus.artifact_relpath, stimulus.image)
            handle.write(json.dumps(stimulus.metadata(), ensure_ascii=True) + "\n")


def build_runner(args: argparse.Namespace, model_id: str) -> Any:
    if args.backend == "oracle":
        return OracleRunner()
    if args.backend == "auto":
        return build_auto_runner(args, model_id)
    if args.backend == "qwen2_5":
        return Qwen25Runner(args, model_id)
    if args.backend == "qwen3":
        return Qwen3Runner(args, model_id)
    if args.backend == "qwen":
        if is_qwen3_model(model_id):
            return Qwen3Runner(args, model_id)
        return Qwen25Runner(args, model_id)
    if args.backend == "llava":
        return LlavaRunner(args, model_id)
    if args.backend == "kimi_vl":
        return KimiVLRunner(args, model_id)
    raise ValueError(f"Unsupported backend: {args.backend}")


def build_auto_runner(args: argparse.Namespace, model_id: str) -> Any:
    if is_qwen_model(model_id):
        if is_qwen3_model(model_id):
            return Qwen3Runner(args, model_id)
        return Qwen25Runner(args, model_id)
    if is_llava_model(model_id):
        return LlavaRunner(args, model_id)
    if is_kimi_vl_model(model_id):
        return KimiVLRunner(args, model_id)
    raise ValueError(
        f"Could not infer backend for {model_id!r}. "
        "Set --backend to qwen2_5, qwen3, llava, or kimi_vl."
    )


def release_runner(runner: Any) -> None:
    close = getattr(runner, "close", None)
    if close is not None:
        close()


class OracleRunner:
    def __call__(self, stimulus: Stimulus, image_path: Path) -> str:
        del image_path
        if stimulus.task_type == "color":
            return json.dumps({"color": stimulus.expected["color"]})
        if stimulus.task_type == "stripes":
            return json.dumps(
                {
                    "orientation": stimulus.expected["orientation"],
                    "stripe_count": stimulus.expected["stripe_count"],
                    "colors": stimulus.expected["colors"],
                }
            )
        raise ValueError(f"Unsupported task type: {stimulus.task_type}")


class Qwen25Runner:
    def __init__(self, args: argparse.Namespace, model_id: str) -> None:
        self.args = args
        self.model_id = model_id
        self.model, self.processor, self.torch = load_qwen25_model_and_processor(args, model_id)

    def __call__(self, stimulus: Stimulus, image_path: Path) -> str:
        messages = build_qwen_messages(
            system_text=system_prompt(),
            user_text=stimulus.prompt,
            image_path=image_path,
        )
        inputs = prepare_inputs(self.processor, messages, self.model, self.torch)
        return generate_response(self.model, self.processor, inputs, self.args, self.torch)

    def close(self) -> None:
        del self.model
        del self.processor
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


class Qwen3Runner:
    def __init__(self, args: argparse.Namespace, model_id: str) -> None:
        self.args = args
        self.model_id = model_id
        self.model, self.processor, self.torch = load_qwen3_model_and_processor(args, model_id)

    def __call__(self, stimulus: Stimulus, image_path: Path) -> str:
        messages = build_qwen_messages(
            system_text=system_prompt(),
            user_text=stimulus.prompt,
            image_path=image_path,
        )
        inputs = prepare_qwen3_inputs(self.processor, messages, self.model, self.torch)
        return generate_response(self.model, self.processor, inputs, self.args, self.torch)

    def close(self) -> None:
        del self.model
        del self.processor
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


class LlavaRunner:
    def __init__(self, args: argparse.Namespace, model_id: str) -> None:
        self.args = args
        self.model_id = model_id
        self.model, self.processor, self.torch = load_llava_model_and_processor(args, model_id)

    def __call__(self, stimulus: Stimulus, image_path: Path) -> str:
        messages = build_llava_messages(
            user_text=build_visual_user_text(stimulus.prompt),
        )
        inputs = prepare_llava_inputs(self.processor, messages, image_path, self.model, self.torch)
        return generate_response(self.model, self.processor, inputs, self.args, self.torch)

    def close(self) -> None:
        del self.model
        del self.processor
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


class KimiVLRunner:
    def __init__(self, args: argparse.Namespace, model_id: str) -> None:
        self.args = args
        self.model_id = model_id
        self.model, self.processor, self.torch = load_kimi_vl_model_and_processor(args, model_id)

    def __call__(self, stimulus: Stimulus, image_path: Path) -> str:
        messages = build_kimi_vl_messages(
            user_text=build_visual_user_text(stimulus.prompt),
            image_path=image_path,
        )
        inputs = prepare_kimi_vl_inputs(self.processor, messages, image_path, self.model, self.torch)
        return generate_response(self.model, self.processor, inputs, self.args, self.torch)

    def close(self) -> None:
        del self.model
        del self.processor
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.empty_cache()


def is_qwen_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen" in normalized and ("vl" in normalized or "vision" in normalized)


def is_qwen3_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen3-vl" in normalized or "qwen3_vl" in normalized


def is_llava_model(model_id: str) -> bool:
    return "llava" in model_id.lower()


def is_kimi_vl_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "kimi-vl" in normalized or "kimi_vl" in normalized


def build_visual_user_text(user_text: str) -> str:
    return f"{system_prompt()}\n\n{user_text}"


def build_qwen_messages(
    *,
    system_text: str,
    user_text: str,
    image_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_llava_messages(*, user_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def build_kimi_vl_messages(
    *,
    user_text: str,
    image_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def load_qwen25_model_and_processor(args: argparse.Namespace, model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision  # noqa: F401
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing open-model dependencies. Run: "
            "python -m pip install -r requirements-open-models.txt"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, processor, torch


def load_llava_model_and_processor(args: argparse.Namespace, model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLaVA dependencies. Run: "
            "python -m pip install -r requirements-open-models.txt"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, processor, torch


def load_kimi_vl_model_and_processor(args: argparse.Namespace, model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing Kimi-VL dependencies. Run: "
            "python -m pip install -r requirements-open-models.txt"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor, torch


def load_qwen3_model_and_processor(args: argparse.Namespace, model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing Qwen3-VL dependencies. Upgrade Transformers, then rerun. "
            "The Qwen3-VL model card recommends: "
            "python -m pip install -U git+https://github.com/huggingface/transformers accelerate"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, processor, torch


def resolve_torch_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def prepare_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    model: Any,
    torch: Any,
) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Missing qwen-vl-utils. Run: python -m pip install -r requirements-open-models.txt"
        ) from exc

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = infer_input_device(model, torch)
    return move_inputs_to_device(inputs, device)


def prepare_qwen3_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    model: Any,
    torch: Any,
) -> dict[str, Any]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = infer_input_device(model, torch)
    return move_inputs_to_device(inputs, device)


def prepare_llava_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    image_path: Path,
    model: Any,
    torch: Any,
) -> Any:
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        images=image,
        text=text,
        padding=True,
        return_tensors="pt",
    )
    device = infer_input_device(model, torch)
    return move_inputs_to_device(inputs, device)


def prepare_kimi_vl_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    image_path: Path,
    model: Any,
    torch: Any,
) -> Any:
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = infer_input_device(model, torch)
    return move_inputs_to_device(inputs, device)


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def infer_input_device(model: Any, torch: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_response(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
) -> str:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
    }
    if args.temperature > 0.0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, prompt_length:]
    outputs = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (outputs[0] if outputs else "").strip()


def build_result_row(
    *,
    model_id: str,
    backend: str,
    stimulus: Stimulus,
    image_path: Path,
    repeat: int,
    raw_output: str,
) -> dict[str, Any]:
    parsed = parse_raw_json(raw_output)
    row: dict[str, Any] = {
        "trial_id": f"{safe_slug(model_id)}__{stimulus.stimulus_id}__r{repeat:02d}",
        "backend": backend,
        "model_id": model_id,
        "stimulus_id": stimulus.stimulus_id,
        "task_type": stimulus.task_type,
        "width": stimulus.width,
        "height": stimulus.height,
        "area_pixels": stimulus.width * stimulus.height,
        "image_path": str(image_path),
        "repeat": repeat,
        "raw_output": raw_output,
        "valid_json": parsed["valid_json"],
        "json_error": parsed["json_error"],
    }
    row.update(flatten_expected(stimulus.expected))
    if stimulus.task_type == "color":
        row.update(evaluate_color(parsed["payload"], stimulus.expected))
    elif stimulus.task_type == "stripes":
        row.update(evaluate_stripes(parsed["payload"], stimulus.expected))
    else:
        raise ValueError(f"Unsupported task type: {stimulus.task_type}")
    return row


def parse_raw_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is not None:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(payload, dict):
            return {"valid_json": True, "payload": payload, "json_error": None}
        last_error = "JSON payload was not an object"
    return {"valid_json": False, "payload": {}, "json_error": last_error}


def flatten_expected(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        f"expected_{key}": json.dumps(value, ensure_ascii=True)
        if isinstance(value, list)
        else value
        for key, value in expected.items()
    }


def evaluate_color(payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    predicted_color = normalize_color(str(payload.get("color", "")))
    correct = predicted_color == expected["color"]
    return {
        "predicted_color": predicted_color or None,
        "color_correct": bool(correct),
        "orientation_correct": None,
        "stripe_count_correct": None,
        "stripe_sequence_correct": None,
        "stripe_set_correct": None,
        "all_correct": bool(correct),
    }


def evaluate_stripes(payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    predicted_orientation = normalize_orientation(payload.get("orientation"))
    predicted_count = parse_count(payload.get("stripe_count"))
    predicted_colors = normalize_color_sequence(payload.get("colors"))
    expected_colors = list(expected["colors"])
    orientation_correct = predicted_orientation == expected["orientation"]
    count_correct = predicted_count == expected["stripe_count"]
    sequence_correct = predicted_colors == expected_colors
    set_correct = sorted(predicted_colors) == sorted(expected_colors)
    all_correct = orientation_correct and count_correct and sequence_correct
    return {
        "predicted_orientation": predicted_orientation,
        "predicted_stripe_count": predicted_count,
        "predicted_colors": json.dumps(predicted_colors, ensure_ascii=True),
        "color_correct": None,
        "orientation_correct": bool(orientation_correct),
        "stripe_count_correct": bool(count_correct),
        "stripe_sequence_correct": bool(sequence_correct),
        "stripe_set_correct": bool(set_correct),
        "all_correct": bool(all_correct),
    }


def normalize_color(value: str) -> str:
    cleaned = str(value).strip().lower()
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    if cleaned in COLOR_ALIASES:
        return COLOR_ALIASES[cleaned]
    spaced = cleaned.replace("_", " ")
    if spaced in COLOR_ALIASES:
        return COLOR_ALIASES[spaced]
    if cleaned in COLOR_MAP:
        return cleaned
    for color in sorted(COLOR_MAP, key=len, reverse=True):
        if color in cleaned or color.replace("_", "") in cleaned:
            return color
    return cleaned


def normalize_orientation(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned.startswith("v") or "left" in cleaned or "right" in cleaned:
        return "vertical"
    if cleaned.startswith("h") or "top" in cleaned or "bottom" in cleaned:
        return "horizontal"
    return cleaned or None


def parse_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    cleaned = str(value).strip().lower()
    if cleaned in WORD_NUMBERS:
        return WORD_NUMBERS[cleaned]
    match = re.search(r"\d+", cleaned)
    if match is not None:
        return int(match.group(0))
    return None


def normalize_color_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_color(str(item)) for item in value]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"\s*(?:,|;|/|\||->| then | and )\s*", text, flags=re.IGNORECASE)
    return [normalize_color(part) for part in parts if part.strip()]


def format_progress(row: dict[str, Any]) -> str:
    if row["task_type"] == "color":
        return (
            f"{row['model_id']} | {row['stimulus_id']} -> {row.get('predicted_color')} "
            f"correct={row['all_correct']}"
        )
    return (
        f"{row['model_id']} | {row['stimulus_id']} -> "
        f"{row.get('predicted_orientation')} {row.get('predicted_stripe_count')} "
        f"{row.get('predicted_colors')} correct={row['all_correct']}"
    )


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], break_threshold: float) -> None:
    with open(out_dir / "results.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    size_summary = summarize_by_size(df)
    size_summary.to_csv(out_dir / "size_summary.csv", index=False)
    summarize_color_details(df).to_csv(out_dir / "color_summary.csv", index=False)
    summarize_color_group_details(df).to_csv(out_dir / "color_group_summary.csv", index=False)
    summarize_stripe_details(df).to_csv(out_dir / "stripe_summary.csv", index=False)
    summarize_breakpoints(size_summary, break_threshold).to_csv(
        out_dir / "breakpoints.csv",
        index=False,
    )
    write_json(
        out_dir / "summary.json",
        {
            "status": "ok",
            "trial_count": len(df),
            "valid_json_count": int(df["valid_json"].sum()) if "valid_json" in df else 0,
            "outputs": [
                "results.csv",
                "results.jsonl",
                "size_summary.csv",
                "color_summary.csv",
                "color_group_summary.csv",
                "stripe_summary.csv",
                "breakpoints.csv",
                "stimuli.jsonl",
            ],
        },
    )


def summarize_by_size(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(["model_id", "task_type", "width", "height"], dropna=False):
        model_id, task_type, width, height = keys
        row = {
            "model_id": model_id,
            "task_type": task_type,
            "width": width,
            "height": height,
            "area_pixels": int(width) * int(height),
            "trial_count": len(group),
            "valid_json_rate": float(group["valid_json"].mean()),
            "all_correct_rate": float(group["all_correct"].mean()),
        }
        if task_type == "color":
            row["color_correct_rate"] = float(group["color_correct"].mean())
        if task_type == "stripes":
            row["orientation_correct_rate"] = float(group["orientation_correct"].mean())
            row["stripe_count_correct_rate"] = float(group["stripe_count_correct"].mean())
            row["stripe_sequence_correct_rate"] = float(group["stripe_sequence_correct"].mean())
            row["stripe_set_correct_rate"] = float(group["stripe_set_correct"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["model_id", "task_type", "area_pixels"],
        ascending=[True, True, False],
    )


def summarize_color_details(df: pd.DataFrame) -> pd.DataFrame:
    color_df = df[df["task_type"] == "color"].copy()
    if color_df.empty:
        return pd.DataFrame()
    return (
        color_df.groupby(["model_id", "width", "height", "expected_color"], dropna=False)
        .agg(
            trial_count=("trial_id", "count"),
            valid_json_rate=("valid_json", "mean"),
            color_correct_rate=("color_correct", "mean"),
            predicted_colors=("predicted_color", unique_json),
        )
        .reset_index()
        .sort_values(["model_id", "width", "height", "expected_color"])
    )


def summarize_color_group_details(df: pd.DataFrame) -> pd.DataFrame:
    color_df = df[df["task_type"] == "color"].copy()
    if color_df.empty or "expected_color_group" not in color_df:
        return pd.DataFrame()
    return (
        color_df.groupby(["model_id", "expected_color_group"], dropna=False)
        .agg(
            trial_count=("trial_id", "count"),
            valid_json_rate=("valid_json", "mean"),
            color_correct_rate=("color_correct", "mean"),
            expected_colors=("expected_color", unique_json),
            predicted_colors=("predicted_color", unique_json),
        )
        .reset_index()
        .sort_values(["model_id", "expected_color_group"])
    )


def summarize_stripe_details(df: pd.DataFrame) -> pd.DataFrame:
    stripe_df = df[df["task_type"] == "stripes"].copy()
    if stripe_df.empty:
        return pd.DataFrame()
    return (
        stripe_df.groupby(
            ["model_id", "width", "height", "expected_pattern_name"],
            dropna=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            valid_json_rate=("valid_json", "mean"),
            all_correct_rate=("all_correct", "mean"),
            orientation_correct_rate=("orientation_correct", "mean"),
            stripe_count_correct_rate=("stripe_count_correct", "mean"),
            stripe_sequence_correct_rate=("stripe_sequence_correct", "mean"),
            predicted_orientations=("predicted_orientation", unique_json),
            predicted_counts=("predicted_stripe_count", unique_json),
            predicted_color_sequences=("predicted_colors", unique_json),
        )
        .reset_index()
        .sort_values(["model_id", "width", "height", "expected_pattern_name"])
    )


def summarize_breakpoints(size_summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if size_summary.empty:
        return pd.DataFrame()
    for keys, group in size_summary.groupby(["model_id", "task_type"], dropna=False):
        model_id, task_type = keys
        ordered = group.sort_values("area_pixels", ascending=False)
        failing = ordered[ordered["all_correct_rate"] < threshold]
        first_bad = failing.iloc[0].to_dict() if not failing.empty else None
        rows.append(
            {
                "model_id": model_id,
                "task_type": task_type,
                "threshold": threshold,
                "largest_to_smallest_first_below_threshold": None
                if first_bad is None
                else f"{int(first_bad['width'])}x{int(first_bad['height'])}",
                "first_below_threshold_accuracy": None
                if first_bad is None
                else float(first_bad["all_correct_rate"]),
                "min_accuracy": float(group["all_correct_rate"].min()),
                "max_accuracy": float(group["all_correct_rate"].max()),
            }
        )
    return pd.DataFrame(rows)


def unique_json(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna()})
    return json.dumps(values, ensure_ascii=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "x"


if __name__ == "__main__":
    main()
