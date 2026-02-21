from __future__ import annotations

from abc import ABC
import inspect
import json
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

ConfigType = Literal["model", "dataset", "backdoor", "wandb", "localfs"]

TConfig = TypeVar("TConfig", bound="AbstractConfig")


class AbstractConfig(ABC):
    """Base class for typed configs loaded from JSON files."""

    config_type: ClassVar[ConfigType]
    name: ClassVar[str]
    json_path: Path | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        if not hasattr(cls, "config_type"):
            raise TypeError(f"{cls.__name__} must define class attribute 'config_type'")
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define class attribute 'name'")

        valid_types = {"model", "dataset", "backdoor", "wandb", "localfs"}
        if cls.config_type not in valid_types:
            valid = ", ".join(sorted(valid_types))
            raise TypeError(f"{cls.__name__}.config_type must be one of: {valid}")

        if not isinstance(cls.name, str) or not cls.name.strip():
            raise TypeError(f"{cls.__name__}.name must be a non-empty string")

    @classmethod
    def from_json(cls: type[TConfig], path: str | Path) -> TConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"{config_path} must contain a JSON object at the top level."
            )

        try:
            config = cls(**raw_config)
            object.__setattr__(config, "json_path", config_path)
            return config
        except TypeError as exc:
            raise ValueError(
                f"Invalid schema for {cls.__name__} loaded from {config_path}: {exc}"
            ) from exc
