from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class BackdoorConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "backdoor"
    name: ClassVar[str] = "whitebox"

    attack_name: str
    p: float
