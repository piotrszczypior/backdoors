from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class WhiteBoxConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "backdoor"
    name: ClassVar[str] = "whitebox"

    placeholder: int
