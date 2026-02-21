from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class WanDbConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "wandb"
    name: ClassVar[str] = "wandb"

    placeholder: int
