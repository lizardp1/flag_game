from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from nnd.flag_game.catalog import COUNTRY_POOLS


class OutputConfig(BaseModel):
    save_crop_images: bool = True
    make_plots: bool = True


class BroadcastFlagGameConfig(BaseModel):
    backend: Literal["openai", "scripted"] = "openai"
    model: str = "gpt-4o"
    agent_models: list[str] | None = None
    prestige_model_label: str = "gpt-5.4"
    comparison_model_label: str = "gpt-4o"
    image_detail: Literal["auto", "low", "high", "original"] = "high"
    N: int = 8
    rounds: int = 10
    H: int = 8
    interaction_m: int = 3
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    max_influential_agents: int = 3
    country_pool: str = "stripe_expanded_24"
    fixed_truth_country: str | None = None
    canvas_width: int = 24
    canvas_height: int = 16
    tile_width: int = 6
    tile_height: int = 4
    observation_overlap: float | None = None
    overlap_search_trials: int = 200
    render_scale: int = 25
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 250
    consensus_threshold: float = 0.9
    polarization_threshold: float = 0.2
    early_stop_round_window: int = 3
    agent_workers: int = 4
    seed_workers: int = 1
    condition_workers: int = 1
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("N")
    @classmethod
    def _check_n(cls, value: int) -> int:
        if value < 1:
            raise ValueError("N must be >= 1")
        return value

    @field_validator("H", "tile_width", "tile_height", "render_scale")
    @classmethod
    def _check_positive_ints(cls, value: int, info: Any) -> int:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        if info.field_name in {"tile_width", "tile_height", "render_scale"} and value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("rounds")
    @classmethod
    def _check_rounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rounds must be >= 1")
        return value

    @field_validator("interaction_m")
    @classmethod
    def _check_interaction_m(cls, value: int) -> int:
        if value not in (1, 2, 3):
            raise ValueError("interaction_m must be one of {1, 2, 3}")
        return value

    @field_validator("social_susceptibility")
    @classmethod
    def _check_social_susceptibility(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("social_susceptibility must be in [0, 1]")
        return value

    @field_validator("observation_overlap")
    @classmethod
    def _check_overlap(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not (0.0 <= value <= 1.0):
            raise ValueError("observation_overlap must be in [0, 1]")
        return value

    @field_validator("country_pool")
    @classmethod
    def _check_country_pool(cls, value: str) -> str:
        if value not in COUNTRY_POOLS:
            valid = ", ".join(sorted(COUNTRY_POOLS))
            raise ValueError(f"country_pool must be one of: {valid}")
        return value

    @field_validator("agent_models")
    @classmethod
    def _check_agent_models(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value]
        if not cleaned:
            return None
        if any(not item for item in cleaned):
            raise ValueError("agent_models entries must be non-empty strings")
        return cleaned

    @field_validator("consensus_threshold")
    @classmethod
    def _check_consensus_threshold(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("consensus_threshold must be in (0, 1]")
        return value

    @field_validator("polarization_threshold")
    @classmethod
    def _check_polarization_threshold(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("polarization_threshold must be in (0, 1]")
        return value

    @field_validator("agent_workers", "seed_workers", "condition_workers", "overlap_search_trials")
    @classmethod
    def _check_worker_counts(cls, value: int, info: Any) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("max_influential_agents")
    @classmethod
    def _check_max_influential_agents(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_influential_agents must be >= 0")
        return value

    @field_validator("prestige_model_label", "comparison_model_label")
    @classmethod
    def _check_model_label(cls, value: str, info: Any) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _post_validate(self) -> "BroadcastFlagGameConfig":
        if self.tile_width > self.canvas_width:
            raise ValueError("tile_width must be <= canvas_width")
        if self.tile_height > self.canvas_height:
            raise ValueError("tile_height must be <= canvas_height")
        if self.agent_models is not None and len(self.agent_models) != self.N:
            raise ValueError(f"agent_models must have length N={self.N}, got {len(self.agent_models)}")
        if self.max_influential_agents >= self.N:
            raise ValueError("max_influential_agents must be < N")
        if self.prestige_model_label == self.comparison_model_label:
            raise ValueError("prestige_model_label and comparison_model_label must differ")
        if self.fixed_truth_country is not None:
            pool = COUNTRY_POOLS[self.country_pool]
            if self.fixed_truth_country not in pool:
                raise ValueError(
                    f"fixed_truth_country={self.fixed_truth_country!r} must be in country_pool={self.country_pool!r}"
                )
        return self


def load_broadcast_flag_game_config(path: Path) -> BroadcastFlagGameConfig:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return BroadcastFlagGameConfig.model_validate(data)


def apply_overrides(config: BroadcastFlagGameConfig, overrides: list[str]) -> BroadcastFlagGameConfig:
    if not overrides:
        return config
    data = config.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(data, key, value)
    return BroadcastFlagGameConfig.model_validate(data)


def _set_nested(config_dict: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = config_dict
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def save_resolved_config(config: BroadcastFlagGameConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
