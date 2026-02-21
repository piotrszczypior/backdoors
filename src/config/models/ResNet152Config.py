from dataclasses import dataclass
from typing import ClassVar

import torchvision.models as models

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class ResNet152Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "resnet152"

    weights: models.ResNet152_Weights = models.ResNet152_Weights.IMAGENET1K_V1
