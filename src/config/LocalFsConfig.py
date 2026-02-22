from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class LocalFsConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "localfs"
    name: ClassVar[str] = "localfs"

    output_dir: str
