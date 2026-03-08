from dataclasses import dataclass
from typing import ClassVar
from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class ObservabilityConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "observability"
    name: ClassVar[str] = "observability"

    collect_images_freq: int = 0
    num_images_to_collect: int = 8
