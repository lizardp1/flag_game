from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nnd.number_game import prompts
from nnd.number_game.domain import filter_candidates
from nnd.number_game.parsing import (
    NumberDecision,
    NumberMessage,
    ParseError,
    parse_number_decision,
    parse_number_message,
)


def _clean_base_url(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip() or None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class NumberGameOpenAICompatibleBackend:
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    debug_dir: Path
    api_base_url: str | None = None
    api_key: str | None = None
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    prompt_number_range: bool = False
    capture_hidden_states: bool = False
    hidden_state_layers: list[int] | None = None
    use_response_format: bool = True

    def __post_init__(self) -> None:
        self.base_url = (
            self.api_base_url
            or _clean_base_url("NND_MODEL_BASE_URL")
            or _clean_base_url("OPENAI_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.active_api_key = (
            self.api_key
            or os.environ.get("NND_MODEL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "local"
        )
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.usage_rows: list[dict[str, Any]] = []
        self.hidden_state_rows: list[dict[str, Any]] = []

    def message(self, *, numbers: list[int], private_clue: str, memory_lines: list[str], m: int) -> NumberMessage:
        text = prompts.interaction_text(
            numbers=numbers,
            private_clue=private_clue,
            memory_lines=memory_lines,
            m=m,
            social_susceptibility=self.social_susceptibility,
            prompt_social_susceptibility=self.prompt_social_susceptibility,
            prompt_number_range=self.prompt_number_range,
        )
        return self._call_with_retries(
            prompts.openai_messages(text),
            lambda response_text: parse_number_message(response_text, allowed_numbers=numbers, m=m),
            retry_builder=lambda exc: prompts.retry_text(numbers=numbers, m=m, error_text=str(exc)),
        )

    def final_decision(
        self,
        *,
        numbers: list[int],
        private_clue: str,
        memory_lines: list[str],
        broadcast_lines: list[str],
        m: int,
        max_influential_agents: int,
        valid_agent_ids: set[int],
    ) -> NumberDecision:
        text = prompts.final_decision_text(
            numbers=numbers,
            private_clue=private_clue,
            memory_lines=memory_lines,
            broadcast_lines=broadcast_lines,
            m=m,
            max_influential_agents=max_influential_agents,
            social_susceptibility=self.social_susceptibility,
            prompt_social_susceptibility=self.prompt_social_susceptibility,
            prompt_number_range=self.prompt_number_range,
        )
        return self._call_with_retries(
            prompts.openai_messages(text),
            lambda response_text: parse_number_decision(
                response_text,
                allowed_numbers=numbers,
                m=m,
                max_influential_agents=max_influential_agents,
                valid_agent_ids=valid_agent_ids,
            ),
            retry_builder=lambda exc: prompts.retry_text(numbers=numbers, m=m, error_text=str(exc), decision=True),
        )

    def organization_decision(
        self,
        *,
        numbers: list[int],
        memory_lines: list[str],
        observer_statement_lines: list[str],
        m: int,
    ) -> NumberMessage:
        text = prompts.organization_decision_text(
            numbers=numbers,
            memory_lines=memory_lines,
            observer_statement_lines=observer_statement_lines,
            m=m,
            prompt_number_range=self.prompt_number_range,
        )
        return self._call_with_retries(
            prompts.openai_messages(text),
            lambda response_text: parse_number_message(response_text, allowed_numbers=numbers, m=m),
            retry_builder=lambda exc: prompts.retry_text(numbers=numbers, m=m, error_text=str(exc)),
        )

    def _call(self, messages: list[dict[str, str]]) -> str:
        extra_body: dict[str, Any] = {}
        if self.capture_hidden_states:
            extra_body["return_hidden_states"] = True
            if self.hidden_state_layers is not None:
                extra_body["hidden_state_layers"] = self.hidden_state_layers
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        request.update(extra_body)
        if self.use_response_format:
            request["response_format"] = {"type": "json_object"}
        response = self._post_chat_completion(request)
        self._record_usage(response)
        self._record_hidden_states(response)
        return _chat_completion_text(response)

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean_payload = {key: value for key, value in payload.items() if value is not None}
        data = json.dumps(clean_payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.active_api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible server returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach OpenAI-compatible model server at {self.base_url}. "
                "Start vLLM/SGLang/LM Studio/Ollama or set api_base_url/NND_MODEL_BASE_URL."
            ) from exc

    def _record_usage(self, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        self.usage_rows.append(
            {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
            }
        )

    def _record_hidden_states(self, response: dict[str, Any]) -> None:
        if not self.capture_hidden_states:
            return
        for key in ("hidden_states", "prompt_hidden_states", "representations"):
            if key in response:
                self.hidden_state_rows.append({"model": self.model, "field": key, "value": response[key]})

    def _call_with_retries(
        self,
        messages: list[dict[str, str]],
        parser: Callable[[str], Any],
        retry_builder: Callable[[ParseError], str],
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
                current_messages = current_messages + [{"role": "user", "content": retry_builder(exc)}]
        raise ParseError("Exceeded retry limit")

    def _call_with_backoff(self, messages: list[dict[str, str]], max_retries: int = 5) -> str:
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                return self._call(messages)
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
        raise RuntimeError("Unreachable backoff state")

    def _write_debug(self, attempts: list[dict[str, Any]]) -> None:
        path = self.debug_dir / f"parse_failure_{int(time.time() * 1000)}.json"
        with open(path, "w") as handle:
            json.dump({"attempts": attempts}, handle, indent=2, default=str)

    def usage_summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_call_count": len(self.usage_rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in self.usage_rows),
            "completion_tokens": sum(row["completion_tokens"] for row in self.usage_rows),
            "total_tokens": sum(row["total_tokens"] for row in self.usage_rows),
            "pricing_known": False,
        }


@dataclass
class TransformersNumberGameBackend(NumberGameOpenAICompatibleBackend):
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "backend=transformers requires torch and transformers. Install them in your Python environment, "
                "or use backend=openai_compatible with a vLLM/SGLang/LM Studio/Ollama server."
            ) from exc

        self.torch = torch
        dtype = self._resolve_torch_dtype(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model_obj.eval()
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.usage_rows: list[dict[str, Any]] = []
        self.hidden_state_rows: list[dict[str, Any]] = []

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.torch_dtype == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype={self.torch_dtype!r}")
        return mapping[self.torch_dtype]

    def _call(self, messages: list[dict[str, str]]) -> str:
        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model_obj, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[-1])

        if self.capture_hidden_states:
            self._capture_prompt_hidden_states(inputs)

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with self.torch.no_grad():
            generated = self.model_obj.generate(**inputs, **generation_kwargs)
        completion_ids = generated[0][input_token_count:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        self.usage_rows.append(
            {
                "model": self.model,
                "prompt_tokens": input_token_count,
                "completion_tokens": int(completion_ids.shape[-1]),
                "total_tokens": input_token_count + int(completion_ids.shape[-1]),
            }
        )
        return text

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"

    def _capture_prompt_hidden_states(self, inputs: dict[str, Any]) -> None:
        with self.torch.no_grad():
            output = self.model_obj(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        layers = self.hidden_state_layers if self.hidden_state_layers is not None else [-1]
        for layer in layers:
            layer_index = layer if layer >= 0 else len(hidden_states) + layer
            if layer_index < 0 or layer_index >= len(hidden_states):
                continue
            tensor = hidden_states[layer_index][0]
            last_vector = tensor[-1].detach().float().cpu().tolist()
            mean_vector = tensor.mean(dim=0).detach().float().cpu().tolist()
            self.hidden_state_rows.append(
                {
                    "model": self.model,
                    "field": "prompt_hidden_state",
                    "layer": layer,
                    "last_token": last_vector,
                    "mean": mean_vector,
                }
            )


@dataclass
class ScriptedNumberGameBackend:
    seed: int
    social_susceptibility: float = 0.5

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def message(self, *, numbers: list[int], private_clue: str, memory_lines: list[str], m: int) -> NumberMessage:
        number = self._choose_number(numbers, private_clue, memory_lines)
        if m == 1:
            return NumberMessage(number=number)
        return NumberMessage(number=number, reason=f"My clue says {private_clue}.")

    def final_decision(
        self,
        *,
        numbers: list[int],
        private_clue: str,
        memory_lines: list[str],
        broadcast_lines: list[str],
        m: int,
        max_influential_agents: int,
        valid_agent_ids: set[int],
    ) -> NumberDecision:
        number = self._choose_number(numbers, private_clue, memory_lines + broadcast_lines)
        influential = tuple(sorted(valid_agent_ids)[:max_influential_agents])
        if m == 1:
            return NumberDecision(number=number, influential_agent_ids=influential)
        return NumberDecision(number=number, influential_agent_ids=influential, reason=f"I combined my clue with {len(broadcast_lines)} broadcasts.")

    def organization_decision(
        self,
        *,
        numbers: list[int],
        memory_lines: list[str],
        observer_statement_lines: list[str],
        m: int,
    ) -> NumberMessage:
        number = self._choose_number(numbers, "", memory_lines + observer_statement_lines)
        if m == 1:
            return NumberMessage(number=number)
        return NumberMessage(number=number, reason="I aggregated the observer statements.")

    def _choose_number(self, numbers: list[int], private_clue: str, evidence_lines: list[str]) -> int:
        private_candidates = filter_candidates(numbers, [private_clue]) if private_clue else numbers
        vote_counts = Counter()
        for line in evidence_lines:
            match = re_match_number(line)
            if match in numbers:
                vote_counts[match] += 1
        scored: list[tuple[float, int]] = []
        for number in numbers:
            private_score = 1.0 if number in private_candidates else 0.0
            social_score = vote_counts[number] / max(sum(vote_counts.values()), 1)
            scored.append(((1.0 - self.social_susceptibility) * private_score + self.social_susceptibility * social_score, -number))
        best_score = max(score for score, _ in scored)
        tied = [numbers[idx] for idx, (score, _) in enumerate(scored) if score == best_score]
        return self.rng.choice(tied)

    def usage_summary(self) -> dict[str, Any]:
        return {"model": "scripted", "api_call_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def re_match_number(line: str) -> int | None:
    import re

    match = re.search(r"\b\d+\b", line)
    return int(match.group(0)) if match else None


def build_backend(
    *,
    backend_name: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    debug_dir: Path,
    seed: int,
    social_susceptibility: float,
    prompt_social_susceptibility: bool,
    prompt_number_range: bool,
    capture_hidden_states: bool,
    hidden_state_layers: list[int] | None,
    use_response_format: bool,
    api_base_url: str | None,
    api_key: str | None,
    trust_remote_code: bool,
    torch_dtype: str,
    device_map: str,
    enable_thinking: bool,
) -> NumberGameOpenAICompatibleBackend | TransformersNumberGameBackend | ScriptedNumberGameBackend:
    if backend_name == "scripted":
        return ScriptedNumberGameBackend(seed=seed, social_susceptibility=social_susceptibility)
    if backend_name == "transformers":
        return TransformersNumberGameBackend(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            debug_dir=debug_dir,
            api_base_url=api_base_url,
            api_key=api_key,
            social_susceptibility=social_susceptibility,
            prompt_social_susceptibility=prompt_social_susceptibility,
            prompt_number_range=prompt_number_range,
            capture_hidden_states=capture_hidden_states,
            hidden_state_layers=hidden_state_layers,
            use_response_format=use_response_format,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            enable_thinking=enable_thinking,
        )
    if backend_name == "openai":
        backend_name = "openai_compatible"
    if backend_name != "openai_compatible":
        raise ValueError(f"Unsupported backend: {backend_name}")
    return NumberGameOpenAICompatibleBackend(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        debug_dir=debug_dir,
        api_base_url=api_base_url,
        api_key=api_key,
        social_susceptibility=social_susceptibility,
        prompt_social_susceptibility=prompt_social_susceptibility,
        prompt_number_range=prompt_number_range,
        capture_hidden_states=capture_hidden_states,
        hidden_state_layers=hidden_state_layers,
        use_response_format=use_response_format,
    )


def _chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""
