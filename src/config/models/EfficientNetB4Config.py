from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class EfficientNetB4Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "efficientnetb4"

    placeholder: int
