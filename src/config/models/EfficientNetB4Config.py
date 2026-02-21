from dataclasses import dataclass
from typing import ClassVar

import torchvision.models as models

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class EfficientNetB4Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "efficientnetb4"

    weights: models.EfficientNet_B4_Weights = (
        models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
