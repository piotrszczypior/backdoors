from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class VitB16Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "vitb16"

    placeholder: int
