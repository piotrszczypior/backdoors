from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class TrainingConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "training"
    name: ClassVar[str] = "training"

    epochs: int = 90
    batch_size: int = 128
    learning_rate_init: float = 0.1
    learning_rate_step: int = 30
    learning_rate_gamma: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    amp: bool = False
