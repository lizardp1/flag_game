#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from nnd.backends.parsing import ParseError
from nnd.flag_game import prompts
from nnd.flag_game.catalog import get_country_pool
from nnd.flag_game.crops import CropBox, crop_image, sample_random_crops, scale_crop_box
from nnd.flag_game.parsing import parse_probe_response
from nnd.flag_game.render import render_flag, save_png


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local Qwen2.5-VL flag-crop smoke test on a RunPod GPU Pod."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "qwen_vl_smoke")
    parser.add_argument("--country-pool", default="stripe_expanded_24")
    parser.add_argument("--truth-country", default="France")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--crop-top", type=int, default=None)
    parser.add_argument("--crop-left", type=int, default=None)
    parser.add_argument("--m", type=int, choices=[1, 2, 3], default=3)
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
        help="Use auto unless you have installed flash-attn and want flash_attention_2.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--activation-summary",
        action="store_true",
        help="Run one extra forward pass with output_hidden_states=True and save shape metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render artifacts and prompt, then exit before importing torch/transformers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    artifact_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stimulus = build_stimulus(args)
    save_png(artifact_dir / "truth_flag.png", stimulus["full_image"])
    save_png(artifact_dir / "agent_crop.png", stimulus["crop_image"])
    (out_dir / "prompt.txt").write_text(stimulus["user_text"])
    write_json(out_dir / "stimulus.json", stimulus["metadata"])

    if args.dry_run:
        write_json(
            out_dir / "result.json",
            {
                "status": "dry_run",
                "model_id": args.model_id,
                "m": args.m,
                "prompt_path": str(out_dir / "prompt.txt"),
                "crop_path": str(artifact_dir / "agent_crop.png"),
                "stimulus": stimulus["metadata"],
            },
        )
        print(f"Dry run complete. Artifacts saved to {out_dir}")
        return

    model, processor, torch = load_model_and_processor(args)
    messages = build_qwen_messages(
        system_text=prompts.system_prompt(),
        user_text=stimulus["user_text"],
        crop_path=(artifact_dir / "agent_crop.png").resolve(),
    )
    inputs = prepare_inputs(processor, messages, model, torch)
    raw_output = generate_response(model, processor, inputs, args, torch)
    parsed = parse_response(raw_output, stimulus["metadata"]["countries"], args.m)
    activation_summary = (
        summarize_hidden_states(model, inputs, torch)
        if args.activation_summary
        else None
    )

    result = {
        "status": "ok",
        "model_id": args.model_id,
        "m": args.m,
        "raw_output": raw_output,
        "parsed": parsed,
        "activation_summary": activation_summary,
        "stimulus": stimulus["metadata"],
        "runtime": runtime_summary(torch),
    }
    write_json(out_dir / "result.json", result)
    print(f"Qwen VLM smoke complete. Artifacts saved to {out_dir}")
    print(f"Raw output: {raw_output}")
    print(f"Parsed: {parsed}")


def build_stimulus(args: argparse.Namespace) -> dict[str, Any]:
    pool = get_country_pool(args.country_pool)
    country_lookup = {flag.country: flag for flag in pool}
    if args.truth_country not in country_lookup:
        valid = ", ".join(sorted(country_lookup))
        raise ValueError(f"truth_country must be in {args.country_pool}: {valid}")
    truth_flag = country_lookup[args.truth_country]
    countries = [flag.country for flag in pool]
    render_width = args.canvas_width * args.render_scale
    render_height = args.canvas_height * args.render_scale
    full_image = render_flag(truth_flag, width=render_width, height=render_height)

    if args.crop_top is None or args.crop_left is None:
        rng = random.Random(args.seed)
        crop_box = sample_random_crops(
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            n_agents=1,
            rng=rng,
        )[0]
    else:
        crop_box = CropBox(
            crop_index=0,
            top=args.crop_top,
            left=args.crop_left,
            height=args.tile_height,
            width=args.tile_width,
        )

    scaled_box = scale_crop_box(crop_box, args.render_scale)
    crop = crop_image(full_image, scaled_box)
    user_text = prompts.probe_text(
        countries=countries,
        memory_lines=[],
        m=args.m,
        social_susceptibility=0.5,
        prompt_social_susceptibility=False,
    )
    return {
        "full_image": full_image,
        "crop_image": crop,
        "user_text": user_text,
        "metadata": {
            "truth_country": truth_flag.country,
            "m": args.m,
            "country_pool": args.country_pool,
            "countries": countries,
            "crop_box": crop_box.to_dict(),
            "pixel_crop_box": scaled_box.to_dict(),
            "canvas": {"width": args.canvas_width, "height": args.canvas_height},
            "tile": {"width": args.tile_width, "height": args.tile_height},
            "render": {"scale": args.render_scale, "width": render_width, "height": render_height},
        },
    }


def build_qwen_messages(*, system_text: str, user_text: str, crop_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_path.as_uri()},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, Any]:
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
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


def prepare_inputs(processor: Any, messages: list[dict[str, Any]], model: Any, torch: Any) -> dict[str, Any]:
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
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def infer_input_device(model: Any, torch: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_response(model: Any, processor: Any, inputs: dict[str, Any], args: argparse.Namespace, torch: Any) -> str:
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


def parse_response(raw_output: str, countries: list[str], m: int) -> dict[str, Any]:
    try:
        message = parse_probe_response(raw_output, countries=countries, m=m)
    except ParseError as exc:
        return {"valid": False, "error": str(exc), "country": None}
    return {"valid": True, **message.to_dict()}


def summarize_hidden_states(model: Any, inputs: dict[str, Any], torch: Any) -> dict[str, Any]:
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        return {"available": False}
    return {
        "available": True,
        "layer_count": len(hidden_states),
        "shapes": [list(state.shape) for state in hidden_states],
        "dtype": str(hidden_states[-1].dtype) if hidden_states else None,
    }


def runtime_summary(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    summary: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        summary.update(
            {
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            }
        )
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
