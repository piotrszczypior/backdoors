from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class WandbConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "wandb"
    name: ClassVar[str] = "wandb"

    entity: str
    project_name: str
    run_id: int = None
