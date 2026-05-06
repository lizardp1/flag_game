from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import anthropic
import httpx
import numpy as np
import openai

from nnd.backends.parsing import ParseError
from nnd.flag_game import open_prompts, prompts
from nnd.flag_game.catalog import COLOR_MAP, FlagSpec, StripeFlag, get_flag
from nnd.flag_game.parsing import (
    InteractionMessage,
    parse_open_country_interaction_response,
    parse_open_country_probe_response,
    parse_interaction_response,
    parse_probe_response,
)
from nnd.flag_game.pricing import MODEL_PRICING_USD_PER_1M_TOKENS
from nnd.flag_game.render import image_to_data_uri
from nnd.net import force_ipv4


def _clean_base_url(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class FlagGameOpenAIBackend:
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    debug_dir: Path
    image_detail: str = "original"
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    prompt_style: str = "closed_country_list"

    def __post_init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        force_ipv4()
        base_url = _clean_base_url("OPENAI_BASE_URL")
        trust_env = _env_flag("NND_HTTP_TRUST_ENV", False)
        http_client = httpx.Client(trust_env=trust_env)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.usage_rows: list[dict[str, Any]] = []

    def prepare_crop(self, crop_image: np.ndarray) -> str:
        return image_to_data_uri(crop_image)

    def interaction(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int,
    ) -> InteractionMessage:
        prompt_module = self._prompt_module()
        messages = prompt_module.openai_multimodal_messages(
            text=prompt_module.interaction_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
                social_susceptibility=self.social_susceptibility,
                prompt_social_susceptibility=self.prompt_social_susceptibility,
            ),
            crop_data_uri=prepared_crop,
            image_detail=self.image_detail,
        )
        return self._call_with_retries(
            messages,
            lambda text: self._parse_interaction_response(text, countries, m),
            retry_builder=lambda exc: prompt_module.interaction_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )

    def probe(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int = 1,
    ) -> InteractionMessage:
        prompt_module = self._prompt_module()
        messages = prompt_module.openai_multimodal_messages(
            text=prompt_module.probe_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
                social_susceptibility=self.social_susceptibility,
                prompt_social_susceptibility=self.prompt_social_susceptibility,
            ),
            crop_data_uri=prepared_crop,
            image_detail=self.image_detail,
        )
        return self._call_with_retries(
            messages,
            lambda text: self._parse_probe_response(text, countries, m),
            retry_builder=lambda exc: prompt_module.probe_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )

    def _prompt_module(self) -> Any:
        if self.prompt_style == "closed_country_list":
            return prompts
        if self.prompt_style == "open_country":
            return open_prompts
        raise ValueError(f"Unsupported prompt_style: {self.prompt_style}")

    def _parse_interaction_response(
        self,
        text: str,
        countries: list[str],
        m: int,
    ) -> InteractionMessage:
        if self.prompt_style == "open_country":
            return parse_open_country_interaction_response(text, countries, m)
        return parse_interaction_response(text, countries, m)

    def _parse_probe_response(
        self,
        text: str,
        countries: list[str],
        m: int,
    ) -> InteractionMessage:
        if self.prompt_style == "open_country":
            return parse_open_country_probe_response(text, countries, m)
        return parse_probe_response(text, countries, m)

    def _call(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        self._record_usage(response)
        content = response.choices[0].message.content
        return content or ""

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(self.model)
        estimated_cost_usd = None
        if pricing is not None:
            estimated_cost_usd = (
                (prompt_tokens / 1_000_000.0) * pricing["input"]
                + (completion_tokens / 1_000_000.0) * pricing["output"]
            )
        self.usage_rows.append(
            {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in self.usage_rows)
        completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in self.usage_rows)
        total_tokens = sum(int(row.get("total_tokens", 0)) for row in self.usage_rows)
        estimated_values = [
            float(row["estimated_cost_usd"])
            for row in self.usage_rows
            if row.get("estimated_cost_usd") is not None
        ]
        return {
            "model": self.model,
            "api_call_count": len(self.usage_rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": sum(estimated_values) if estimated_values else None,
            "pricing_known": self.model in MODEL_PRICING_USD_PER_1M_TOKENS,
        }

    def _call_with_retries(
        self,
        messages: list[dict[str, Any]],
        parser: Callable[[str], Any],
        retry_builder: Callable[[ParseError], str] | None = None,
        max_retries: int = 2,
    ) -> Any:
        attempts: list[dict[str, Any]] = []
        current_messages = list(messages)
        for attempt in range(max_retries + 1):
            response_text = self._call_with_backoff(current_messages)
            attempts.append({"messages": current_messages, "response": response_text})
            try:
                return parser(response_text)
            except ParseError as exc:
                if attempt >= max_retries:
                    self._write_debug(attempts)
                    raise
                retry_text = (
                    retry_builder(exc)
                    if retry_builder is not None
                    else "Format error. Respond with STRICT JSON only matching the required schema. No extra text."
                )
                current_messages = list(current_messages) + [
                    {
                        "role": "user",
                        "content": retry_text,
                    }
                ]
        self._write_debug(attempts)
        raise ParseError("Exceeded retry limit")

    def _call_with_backoff(
        self,
        messages: list[dict[str, Any]],
        max_retries: int = 8,
        base_delay: float = 5.0,
        max_delay: float = 60.0,
    ) -> str:
        delay = base_delay
        for attempt in range(max_retries + 1):
            try:
                return self._call(messages)
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                time.sleep(delay)
                delay = min(delay * 2.0, max_delay)
        raise RuntimeError("Unreachable backoff state")

    def _write_debug(self, attempts: list[dict[str, Any]]) -> None:
        timestamp = int(time.time() * 1000)
        path = self.debug_dir / f"parse_failure_{timestamp}.json"
        with open(path, "w") as handle:
            json.dump({"attempts": attempts}, handle, indent=2, default=str)


@dataclass
class FlagGameAnthropicBackend:
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    debug_dir: Path
    image_detail: str = "original"
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    prompt_style: str = "closed_country_list"

    def __post_init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        force_ipv4()
        base_url = _clean_base_url("ANTHROPIC_BASE_URL") or _clean_base_url("ANTHROPIC_API_URL")
        trust_env = _env_flag("NND_HTTP_TRUST_ENV", False)
        http_client = httpx.Client(trust_env=trust_env)
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, http_client=http_client)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.usage_rows: list[dict[str, Any]] = []

    def prepare_crop(self, crop_image: np.ndarray) -> str:
        return image_to_data_uri(crop_image)

    def interaction(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int,
    ) -> InteractionMessage:
        prompt_module = self._prompt_module()
        request_parts = self._anthropic_multimodal_messages(
            text=prompt_module.interaction_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
                social_susceptibility=self.social_susceptibility,
                prompt_social_susceptibility=self.prompt_social_susceptibility,
            ),
            crop_data_uri=prepared_crop,
        )
        return self._call_with_retries(
            request_parts,
            lambda text: self._parse_interaction_response(text, countries, m),
            retry_builder=lambda exc: prompt_module.interaction_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )

    def probe(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int = 1,
    ) -> InteractionMessage:
        prompt_module = self._prompt_module()
        request_parts = self._anthropic_multimodal_messages(
            text=prompt_module.probe_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
                social_susceptibility=self.social_susceptibility,
                prompt_social_susceptibility=self.prompt_social_susceptibility,
            ),
            crop_data_uri=prepared_crop,
        )
        return self._call_with_retries(
            request_parts,
            lambda text: self._parse_probe_response(text, countries, m),
            retry_builder=lambda exc: prompt_module.probe_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )

    def _prompt_module(self) -> Any:
        if self.prompt_style == "closed_country_list":
            return prompts
        if self.prompt_style == "open_country":
            return open_prompts
        raise ValueError(f"Unsupported prompt_style: {self.prompt_style}")

    def _parse_interaction_response(
        self,
        text: str,
        countries: list[str],
        m: int,
    ) -> InteractionMessage:
        if self.prompt_style == "open_country":
            return parse_open_country_interaction_response(text, countries, m)
        return parse_interaction_response(text, countries, m)

    def _parse_probe_response(
        self,
        text: str,
        countries: list[str],
        m: int,
    ) -> InteractionMessage:
        if self.prompt_style == "open_country":
            return parse_open_country_probe_response(text, countries, m)
        return parse_probe_response(text, countries, m)

    def _anthropic_multimodal_messages(
        self,
        *,
        text: str,
        crop_data_uri: str,
    ) -> dict[str, Any]:
        media_type, payload = _split_data_uri(crop_data_uri)
        return {
            "system": self._prompt_module().system_prompt(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": payload,
                            },
                        },
                    ],
                }
            ],
        }

    def _call(self, request_parts: dict[str, Any]) -> str:
        request = {
            "model": self.model,
            "system": request_parts["system"],
            "messages": request_parts["messages"],
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None and self.top_p != 1.0:
            request["top_p"] = self.top_p
        else:
            request["temperature"] = self.temperature
        response = self.client.messages.create(**request)
        self._record_usage(response)
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
        return "".join(parts)

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(self.model)
        estimated_cost_usd = None
        if pricing is not None:
            estimated_cost_usd = (
                (prompt_tokens / 1_000_000.0) * pricing["input"]
                + (completion_tokens / 1_000_000.0) * pricing["output"]
            )
        self.usage_rows.append(
            {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in self.usage_rows)
        completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in self.usage_rows)
        total_tokens = sum(int(row.get("total_tokens", 0)) for row in self.usage_rows)
        estimated_values = [
            float(row["estimated_cost_usd"])
            for row in self.usage_rows
            if row.get("estimated_cost_usd") is not None
        ]
        return {
            "model": self.model,
            "api_call_count": len(self.usage_rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": sum(estimated_values) if estimated_values else None,
            "pricing_known": self.model in MODEL_PRICING_USD_PER_1M_TOKENS,
        }

    def _call_with_retries(
        self,
        request_parts: dict[str, Any],
        parser: Callable[[str], Any],
        retry_builder: Callable[[ParseError], str] | None = None,
        max_retries: int = 2,
    ) -> Any:
        attempts: list[dict[str, Any]] = []
        current_messages = list(request_parts["messages"])
        for attempt in range(max_retries + 1):
            current_request = {**request_parts, "messages": current_messages}
            response_text = self._call_with_backoff(current_request)
            attempts.append({"messages": current_messages, "response": response_text})
            try:
                return parser(response_text)
            except ParseError as exc:
                if attempt >= max_retries:
                    self._write_debug(attempts)
                    raise
                retry_text = (
                    retry_builder(exc)
                    if retry_builder is not None
                    else "Format error. Respond with STRICT JSON only matching the required schema. No extra text."
                )
                current_messages = current_messages + [
                    {
                        "role": "user",
                        "content": retry_text,
                    }
                ]
        self._write_debug(attempts)
        raise ParseError("Exceeded retry limit")

    def _call_with_backoff(
        self,
        request_parts: dict[str, Any],
        max_retries: int = 8,
        base_delay: float = 5.0,
        max_delay: float = 60.0,
    ) -> str:
        delay = base_delay
        for attempt in range(max_retries + 1):
            try:
                return self._call(request_parts)
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                time.sleep(delay)
                delay = min(delay * 2.0, max_delay)
        raise RuntimeError("Unreachable backoff state")

    def _write_debug(self, attempts: list[dict[str, Any]]) -> None:
        timestamp = int(time.time() * 1000)
        path = self.debug_dir / f"parse_failure_{timestamp}.json"
        with open(path, "w") as handle:
            json.dump({"attempts": attempts}, handle, indent=2, default=str)


@dataclass
class ScriptedFlagGameBackend:
    seed: int = 0
    social_susceptibility: float = 0.5
    country_lookup: dict[str, FlagSpec] | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def prepare_crop(self, crop_image: np.ndarray) -> np.ndarray:
        return crop_image

    def interaction(
        self,
        *,
        countries: list[str],
        prepared_crop: np.ndarray,
        memory_lines: list[str],
        m: int,
    ) -> InteractionMessage:
        return self.probe(
            countries=countries,
            prepared_crop=prepared_crop,
            memory_lines=memory_lines,
            m=m,
        )

    def probe(
        self,
        *,
        countries: list[str],
        prepared_crop: np.ndarray,
        memory_lines: list[str],
        m: int = 1,
    ) -> InteractionMessage:
        scores = {country: self._country_score(country, prepared_crop, memory_lines) for country in countries}
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_country = ordered[0][0]
        if m == 1:
            return InteractionMessage(country=best_country)
        if m == 2:
            clue = self._short_clue(best_country, prepared_crop)
            return InteractionMessage(country=best_country, clue=clue)
        reason = self._reason(best_country, prepared_crop, memory_lines)
        return InteractionMessage(country=best_country, reason=reason)

    def _country_score(self, country: str, crop_image: np.ndarray, memory_lines: list[str]) -> float:
        if self.country_lookup is not None and country in self.country_lookup:
            flag = self.country_lookup[country]
        else:
            flag = get_flag(country)
        memory_votes = Counter()
        for line in memory_lines:
            match = re.match(r"([^|]+)", line)
            if match:
                memory_votes[match.group(1).strip()] += 1
        total_votes = sum(memory_votes.values())
        social_score = 0.0 if total_votes == 0 else memory_votes.get(country, 0) / float(total_votes)
        if not isinstance(flag, StripeFlag):
            return self.social_susceptibility * (4.0 * social_score)

        visible_colors = self._visible_color_names(crop_image)
        orientation = self._infer_orientation(crop_image)
        private_score = 0.0

        if orientation is not None:
            private_score += 1.0 if orientation == flag.orientation else -1.0

        for color in visible_colors:
            if color in flag.colors or color == flag.triangle_color:
                private_score += 1.5
            else:
                private_score -= 0.5

        ordered_colors = self._ordered_visible_colors(crop_image, orientation)
        if ordered_colors:
            if self._is_contiguous_subsequence(flag.colors, ordered_colors):
                private_score += 2.0
            else:
                private_score -= 0.5

        return (1.0 - self.social_susceptibility) * private_score + self.social_susceptibility * (4.0 * social_score)

    def _visible_color_names(self, crop_image: np.ndarray) -> list[str]:
        unique = {tuple(int(value) for value in pixel) for pixel in crop_image.reshape(-1, 3)}
        inverse = {rgb: name for name, rgb in COLOR_MAP.items()}
        names = [inverse[rgb] for rgb in unique if rgb in inverse]
        return sorted(names)

    def _infer_orientation(self, crop_image: np.ndarray) -> str | None:
        if crop_image.shape[0] < 2 or crop_image.shape[1] < 2:
            return None
        horizontal_change = float(np.mean(np.any(crop_image[:, 1:, :] != crop_image[:, :-1, :], axis=2)))
        vertical_change = float(np.mean(np.any(crop_image[1:, :, :] != crop_image[:-1, :, :], axis=2)))
        if horizontal_change > vertical_change * 1.5:
            return "vertical"
        if vertical_change > horizontal_change * 1.5:
            return "horizontal"
        return None

    def _ordered_visible_colors(self, crop_image: np.ndarray, orientation: str | None) -> tuple[str, ...]:
        inverse = {rgb: name for name, rgb in COLOR_MAP.items()}
        names: list[str] = []
        if orientation == "vertical":
            for col in range(crop_image.shape[1]):
                rgb = tuple(int(value) for value in crop_image[:, col, :].mean(axis=0).round().astype(int))
                if rgb in inverse and (not names or names[-1] != inverse[rgb]):
                    names.append(inverse[rgb])
        elif orientation == "horizontal":
            for row in range(crop_image.shape[0]):
                rgb = tuple(int(value) for value in crop_image[row, :, :].mean(axis=0).round().astype(int))
                if rgb in inverse and (not names or names[-1] != inverse[rgb]):
                    names.append(inverse[rgb])
        return tuple(names)

    def _is_contiguous_subsequence(self, full: tuple[str, ...], subseq: tuple[str, ...]) -> bool:
        if not subseq:
            return False
        size = len(subseq)
        for start in range(0, len(full) - size + 1):
            if full[start : start + size] == subseq:
                return True
        return False

    def _short_clue(self, country: str, crop_image: np.ndarray) -> str:
        colors = self._visible_color_names(crop_image)
        orientation = self._infer_orientation(crop_image)
        parts = list(colors[:3])
        if orientation:
            parts.append(orientation)
        if not parts:
            parts = [country]
        return " ".join(parts)

    def _reason(self, country: str, crop_image: np.ndarray, memory_lines: list[str]) -> str:
        colors = self._visible_color_names(crop_image)
        orientation = self._infer_orientation(crop_image)
        if memory_lines:
            partner_hint = memory_lines[-1].split("|", 1)[0].strip()
            if colors and orientation:
                return f"I see {' and '.join(colors[:3])} with a {orientation} stripe pattern, and the latest message pointed toward {partner_hint}."
            return f"My crop is limited, so I am leaning on the latest message toward {partner_hint}."
        if colors and orientation:
            return f"I see {' and '.join(colors[:3])} with what looks like a {orientation} stripe pattern."
        if colors:
            return f"I can only see {' and '.join(colors[:3])} in my crop."
        return f"I am choosing {country} from the limited crop evidence."

    def usage_summary(self) -> dict[str, Any]:
        return {
            "model": "scripted",
            "api_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "pricing_known": True,
        }


def _split_data_uri(data_uri: str) -> tuple[str, str]:
    prefix, separator, payload = data_uri.partition(",")
    if separator != "," or not prefix.startswith("data:") or ";base64" not in prefix:
        raise ValueError("Expected a base64 data URI for an Anthropic image block")
    media_type = prefix.removeprefix("data:").split(";", 1)[0]
    if not media_type:
        raise ValueError("Data URI is missing a media type")
    return media_type, payload


def build_backend(
    *,
    backend_name: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    debug_dir: Path,
    image_detail: str,
    seed: int,
    social_susceptibility: float,
    prompt_social_susceptibility: bool,
    prompt_style: str = "closed_country_list",
    country_lookup: dict[str, FlagSpec] | None = None,
) -> FlagGameOpenAIBackend | FlagGameAnthropicBackend | ScriptedFlagGameBackend:
    if backend_name == "scripted":
        return ScriptedFlagGameBackend(
            seed=seed,
            social_susceptibility=social_susceptibility,
            country_lookup=country_lookup,
        )
    if backend_name == "anthropic":
        return FlagGameAnthropicBackend(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            debug_dir=debug_dir,
            image_detail=image_detail,
            social_susceptibility=social_susceptibility,
            prompt_social_susceptibility=prompt_social_susceptibility,
            prompt_style=prompt_style,
        )
    return FlagGameOpenAIBackend(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        debug_dir=debug_dir,
        image_detail=image_detail,
        social_susceptibility=social_susceptibility,
        prompt_social_susceptibility=prompt_social_susceptibility,
        prompt_style=prompt_style,
    )
