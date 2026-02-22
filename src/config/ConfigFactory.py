from __future__ import annotations

from pathlib import Path
from typing import TypeVar, Optional

from config.abstract.AbstractConfig import AbstractConfig, ConfigType

TConfig = TypeVar("TConfig", bound=AbstractConfig)


class ConfigFactory:
    """Factory for loading configuration files and mapping them to typed objects."""

    _registry: dict[tuple[ConfigType, str], type[AbstractConfig]] = {}

    @classmethod
    def register(cls, config_cls: type[AbstractConfig]) -> type[AbstractConfig]:
        """Registers a configuration class in the factory."""
        if not hasattr(config_cls, "config_type") or not hasattr(config_cls, "name"):
            raise TypeError(
                f"Class {config_cls.__name__} must define 'config_type' and 'name' attributes."
            )
        cls._registry[(config_cls.config_type, config_cls.name)] = config_cls
        return config_cls

    @classmethod
    def get_config_class(
        cls, config_type: ConfigType, name: str
    ) -> type[AbstractConfig]:
        """Returns the configuration class registered for the given type and name."""
        try:
            return cls._registry[(config_type, name)]
        except KeyError:
            available = ", ".join(
                f"({ct}, {n})" for (ct, n) in cls._registry.keys() if ct == config_type
            )
            raise ValueError(
                f"No configuration class registered for type '{config_type}' and name '{name}'. "
                f"Available for type '{config_type}': {available}"
            )

    @classmethod
    def load(
        cls,
        config_type: ConfigType,
        config_path: Path | str,
        name: Optional[str] = None,
    ) -> AbstractConfig:
        """Loads a configuration object from a standard JSON file path."""
        config_cls = cls.get_config_class(
            config_type=config_type, name=name if name is not None else config_type
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        return config_cls.from_json(config_path)
