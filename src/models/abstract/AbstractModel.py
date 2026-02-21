from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Any, ClassVar, Generic, Self, TypeVar

from config.abstract.AbstractConfig import AbstractConfig

TConfig = TypeVar("TConfig", bound=AbstractConfig)


class AbstractModel(ABC, Generic[TConfig]):
    """
    Base class for model wrappers that are instantiated from concrete configs.
    """

    config_cls: ClassVar[type[TConfig]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        if not hasattr(cls, "config_cls"):
            raise TypeError(f"{cls.__name__} must define class attribute 'config_cls'")

        if not issubclass(cls.config_cls, AbstractConfig):
            raise TypeError(
                f"{cls.__name__}.config_cls must inherit from AbstractConfig"
            )

        if cls.config_cls.config_type != "model":
            raise TypeError(
                f"{cls.__name__}.config_cls.config_type must be 'model', got "
                f"{cls.config_cls.config_type!r}"
            )

    def __init__(self, config: TConfig) -> None:
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.__class__.__name__} requires config type "
                f"{self.config_cls.__name__}, got {type(config).__name__}"
            )

        self.config: TConfig = config

    @classmethod
    def from_config(cls: type[Self], config: TConfig) -> Self:
        """
        Create a concrete model instance from a concrete AbstractConfig subclass.
        """
        if not isinstance(config, cls.config_cls):
            raise TypeError(
                f"{cls.__name__}.from_config requires {cls.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        return cls(config)

    @abstractmethod
    def build(self):
        """
        Build and return the concrete model instance.
        """
