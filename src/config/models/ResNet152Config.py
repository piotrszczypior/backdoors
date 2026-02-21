from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class ResNet152Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "resnet152"

    placeholder: int
