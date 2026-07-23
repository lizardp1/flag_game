from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class OutputConfig:
    make_plots: bool = False


@dataclass
class NumberGameConfig:
    backend: Literal["openai", "openai_compatible", "transformers", "scripted"] = "scripted"
    model: str = "gpt-4o-mini"
    agent_models: list[str] | None = None
    protocol: Literal["pairwise", "broadcast", "org"] = "pairwise"
    N: int = 8
    T: int = 160
    rounds: int = 10
    H: int = 8
    interaction_m: int = 1
    probe_every: int | None = None
    min_number: int = 1
    max_number: int = 30
    prompt_number_range: bool = False
    fixed_truth_number: int | None = None
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    max_influential_agents: int = 3
    aggregator_agent_id: int = 0
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 160
    consensus_threshold: float = 0.9
    early_stop_window: int = 3
    agent_workers: int = 4
    seed_workers: int = 1
    condition_workers: int = 1
    capture_hidden_states: bool = False
    hidden_state_layers: list[int] | None = None
    use_response_format: bool = True
    api_base_url: str | None = None
    api_key: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    enable_thinking: bool = False
    output: OutputConfig = field(default_factory=OutputConfig)

    def model_copy(self, *, update: dict[str, Any] | None = None) -> "NumberGameConfig":
        data = asdict(self)
        if update:
            data.update(update)
        return _config_from_dict(data)


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text in ("null", "None", "~"):
        return None
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text.strip("\"'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        data[key] = _parse_scalar(value)
    return data


def _config_from_dict(data: dict[str, Any]) -> NumberGameConfig:
    clean = dict(data)
    output = clean.get("output")
    if isinstance(output, dict):
        clean["output"] = OutputConfig(**output)
    elif output is None or isinstance(output, OutputConfig):
        clean["output"] = output or OutputConfig()
    known = {field.name for field in NumberGameConfig.__dataclass_fields__.values()}
    config = NumberGameConfig(**{key: value for key, value in clean.items() if key in known})
    _validate(config)
    return config


def _validate(config: NumberGameConfig) -> None:
    if config.backend not in ("openai", "openai_compatible", "transformers", "scripted"):
        raise ValueError("backend must be openai, openai_compatible, transformers, or scripted")
    if config.protocol not in ("pairwise", "broadcast", "org"):
        raise ValueError("protocol must be pairwise, broadcast, or org")
    if config.N < 1:
        raise ValueError("N must be >= 1")
    if config.T < 0 or config.H < 0:
        raise ValueError("T and H must be >= 0")
    if config.rounds < 1:
        raise ValueError("rounds must be >= 1")
    if config.probe_every is not None and config.probe_every < 1:
        raise ValueError("probe_every must be >= 1 when set")
    if config.interaction_m not in (1, 3):
        raise ValueError("number game interaction_m must be 1 or 3")
    if config.max_number < config.min_number:
        raise ValueError("max_number must be >= min_number")
    if config.fixed_truth_number is not None and not (config.min_number <= config.fixed_truth_number <= config.max_number):
        raise ValueError("fixed_truth_number must lie inside [min_number, max_number]")
    if not (0.0 <= config.social_susceptibility <= 1.0):
        raise ValueError("social_susceptibility must be in [0, 1]")
    total_agents = config.N + 1 if config.protocol == "org" else config.N
    if config.agent_models is not None and len(config.agent_models) != total_agents:
        raise ValueError(f"agent_models must have length {total_agents}, got {len(config.agent_models)}")
    if config.protocol == "org" and config.aggregator_agent_id >= total_agents:
        raise ValueError("aggregator_agent_id must be < N + 1")
    if config.max_influential_agents >= max(config.N, 1):
        raise ValueError("max_influential_agents must be < N")


def load_number_game_config(path: Path) -> NumberGameConfig:
    return _config_from_dict(_load_simple_yaml(path))


def apply_overrides(config: NumberGameConfig, overrides: list[str]) -> NumberGameConfig:
    data = asdict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = _parse_scalar(raw_value)
    return _config_from_dict(data)


def save_resolved_config(config: NumberGameConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as handle:
        for key, value in asdict(config).items():
            handle.write(f"{key}: {value!r}\n")
