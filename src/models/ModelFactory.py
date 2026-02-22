from __future__ import annotations

from typing import Any, TypeVar
from pathlib import Path
import importlib
import pkgutil

from config.abstract.AbstractConfig import AbstractConfig
from models.abstract.AbstractModel import AbstractModel
from output.Log import Log

TModel = TypeVar("TModel", bound=AbstractModel)
log = Log.for_source(__name__)


class ModelFactory:
    """Factory for building models from their configurations."""

    _registry: dict[str, type[AbstractModel]] = {}

    @classmethod
    def register(cls, model_cls: type[AbstractModel]) -> type[AbstractModel]:
        """Registers a model class in the factory."""
        if not hasattr(model_cls, "config_cls"):
            raise TypeError(
                f"Class {model_cls.__name__} must define 'config_cls' attribute."
            )
        cls._registry[model_cls.config_cls.name] = model_cls
        return model_cls

    @classmethod
    def get_model_class(cls, model_name: str) -> type[AbstractModel]:
        """Returns the model class registered for the given model name."""
        try:
            return cls._registry[model_name]
        except KeyError:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"No model class registered for name '{model_name}'. "
                f"Available models: {available}"
            )

    @classmethod
    def build(cls, model_config: AbstractConfig) -> Any:
        """Builds and returns a concrete model instance based on configuration."""
        log.information("model_factory_build_started", model_name=model_config.name)
        model_cls = cls.get_model_class(model_config.name)
        model_wrapper = model_cls.from_config(model_config)
        model = model_wrapper.build()
        log.information(
            "model_factory_build_completed",
            model_name=model_config.name,
            model_class=type(model).__name__,
        )
        return model

    @classmethod
    def discover_models(cls):
        """Discovers and registers all model classes in the models package."""
        models_root = Path(__file__).parent
        package = __package__ + "." if __package__ is not None else ""

        for _, name, is_pkg in pkgutil.walk_packages([str(models_root)], package):
            if not is_pkg and name.endswith("Model"):
                try:
                    importlib.import_module(name)
                except (ImportError, TypeError):
                    pass


ModelFactory.discover_models()
