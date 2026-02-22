from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class BackdoorConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "backdoor"
    name: ClassVar[str] = "backdoor"

    poison_rate: float
    trigger_type: str  # e.g., "white_box", "gaussian_noise"
    target_mapping: str  # e.g., "all_to_one", "source_to_target"
    target_class: int
    selector_type: str = "random_selector"  # e.g., "random_selector", "source_selector"
    seed: int = 42
    source_classes: list[int] = None
