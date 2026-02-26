from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class BackdoorConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "backdoor"
    name: ClassVar[str] = "backdoor"

    poison_rate: float
    trigger_type: str
    target_mapping: str
    target_class: int
    attack_mode: str = "dirty_label"
    selector_type: str = "random_selector"
    seed: int = 42
    source_classes: list[int] = None
