from dataclasses import dataclass, field
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class WandbConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "wandb"
    name: ClassVar[str] = "wandb"

    entity: str
    project: str
    notes: str
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
