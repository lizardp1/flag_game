from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from nnd.flag_game.catalog import COUNTRY_POOLS


class OutputConfig(BaseModel):
    save_crop_images: bool = True
    make_plots: bool = True


class FlagGameConfig(BaseModel):
    backend: Literal["openai", "anthropic", "scripted"] = "openai"
    model: str = "gpt-4o-mini"
    agent_models: list[str] | None = None
    speaker_weights: list[float] | None = None
    image_detail: Literal["auto", "low", "high", "original"] = "original"
    N: int = 8
    T: int = 160
    H: int = 8
    interaction_m: int = 1
    social_susceptibility: float = 0.5
    prompt_social_susceptibility: bool = True
    prompt_style: Literal["closed_country_list", "open_country"] = "closed_country_list"
    country_pool: str = "stripe_expanded_24"
    fixed_truth_country: str | None = None
    canvas_width: int = 24
    canvas_height: int = 16
    tile_width: int = 3
    tile_height: int = 4
    observation_overlap: float | None = None
    observation_overlap_mode: Literal[
        "duplicated_redundancy",
        "legacy_clustered",
        "distinct_geometric",
    ] = "duplicated_redundancy"
    overlap_search_trials: int = 200
    engineered_crop_agent_id: int | None = None
    engineered_crop_preference: Literal["best", "worst"] | None = None
    render_scale: int = 1
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 200
    probe_every: int | None = None
    early_stop_probe_window: int = 5
    consensus_threshold: float = 0.9
    polarization_threshold: float = 0.2
    disable_memory_updates: bool = False
    probe_workers: int = 4
    seed_workers: int = 1
    condition_workers: int = 1
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("N")
    @classmethod
    def _check_n(cls, value: int) -> int:
        if value < 1:
            raise ValueError("N must be >= 1")
        return value

    @field_validator("T")
    @classmethod
    def _check_t(cls, value: int) -> int:
        if value < 0:
            raise ValueError("T must be >= 0")
        return value

    @field_validator("H")
    @classmethod
    def _check_h(cls, value: int) -> int:
        if value < 0:
            raise ValueError("H must be >= 0")
        return value

    @field_validator("tile_width", "tile_height", "render_scale")
    @classmethod
    def _check_tile_dims(cls, value: int) -> int:
        if value < 1:
            raise ValueError("tile dimensions and render_scale must be >= 1")
        return value

    @field_validator("interaction_m")
    @classmethod
    def _check_interaction_m(cls, value: int) -> int:
        if value not in (1, 2, 3):
            raise ValueError("interaction_m must be one of {1, 2, 3}")
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

    @field_validator("speaker_weights")
    @classmethod
    def _check_speaker_weights(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        cleaned = [float(item) for item in value]
        if not cleaned:
            return None
        if any(item <= 0.0 for item in cleaned):
            raise ValueError("speaker_weights entries must be > 0")
        return cleaned

    @field_validator("social_susceptibility")
    @classmethod
    def _check_social_susceptibility(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("social_susceptibility must be in [0, 1]")
        return value

    @field_validator("observation_overlap")
    @classmethod
    def _check_observation_overlap(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not (0.0 <= value <= 1.0):
            raise ValueError("observation_overlap must be in [0, 1]")
        return value

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

    @field_validator("probe_workers", "seed_workers", "condition_workers")
    @classmethod
    def _check_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("worker counts must be >= 1")
        return value

    @field_validator("overlap_search_trials")
    @classmethod
    def _check_overlap_search_trials(cls, value: int) -> int:
        if value < 1:
            raise ValueError("overlap_search_trials must be >= 1")
        return value

    @field_validator("engineered_crop_agent_id")
    @classmethod
    def _check_engineered_crop_agent_id(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("engineered_crop_agent_id must be >= 0")
        return value

    @field_validator("early_stop_probe_window")
    @classmethod
    def _check_early_stop_probe_window(cls, value: int) -> int:
        if value < 0:
            raise ValueError("early_stop_probe_window must be >= 0")
        return value

    @model_validator(mode="after")
    def _post_validate(self) -> "FlagGameConfig":
        if self.tile_width > self.canvas_width:
            raise ValueError("tile_width must be <= canvas_width")
        if self.tile_height > self.canvas_height:
            raise ValueError("tile_height must be <= canvas_height")
        if self.probe_every is None:
            self.probe_every = max(self.N // 2, 1)
        if self.probe_every < 1:
            raise ValueError("probe_every must be >= 1")
        if self.agent_models is not None and len(self.agent_models) != self.N:
            raise ValueError(f"agent_models must have length N={self.N}, got {len(self.agent_models)}")
        if self.speaker_weights is not None and len(self.speaker_weights) != self.N:
            raise ValueError(f"speaker_weights must have length N={self.N}, got {len(self.speaker_weights)}")
        if self.engineered_crop_agent_id is None and self.engineered_crop_preference is not None:
            raise ValueError("engineered_crop_preference requires engineered_crop_agent_id")
        if self.engineered_crop_agent_id is not None and self.engineered_crop_preference is None:
            raise ValueError("engineered_crop_agent_id requires engineered_crop_preference")
        if self.engineered_crop_agent_id is not None and self.engineered_crop_agent_id >= self.N:
            raise ValueError(
                f"engineered_crop_agent_id must be < N={self.N}, got {self.engineered_crop_agent_id}"
            )
        if self.fixed_truth_country is not None:
            pool = COUNTRY_POOLS[self.country_pool]
            if self.fixed_truth_country not in pool:
                raise ValueError(
                    f"fixed_truth_country={self.fixed_truth_country!r} must be in country_pool={self.country_pool!r}"
                )
        return self


def load_flag_game_config(path: Path) -> FlagGameConfig:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return FlagGameConfig.model_validate(data)


def apply_overrides(config: FlagGameConfig, overrides: list[str]) -> FlagGameConfig:
    if not overrides:
        return config
    data = config.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(data, key, value)
    return FlagGameConfig.model_validate(data)


def _set_nested(config_dict: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = config_dict
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def save_resolved_config(config: FlagGameConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config_resolved.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump(config.model_dump(), handle, sort_keys=False)
