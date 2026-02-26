from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class DatasetConfig(AbstractConfig):
    config_type: ClassVar[ConfigType] = "dataset"
    name: ClassVar[str] = "dataset"

    data_path: str
    num_workers: int
