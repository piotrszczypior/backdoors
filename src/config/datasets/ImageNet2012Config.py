from dataclasses import dataclass
from typing import ClassVar

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class ImageNet2012Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "dataset"
    name: ClassVar[str] = "imagenet2012"

    placeholder: int
