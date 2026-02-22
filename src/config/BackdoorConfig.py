from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class BackdoorConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "backdoor"
    name: ClassVar[str] = "backdoor"

    poison_rate: float
    attack_mode: str
    trigger_type: str
    attack_mode: str
    target_mapping: str
    target_class: int
    