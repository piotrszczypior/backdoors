from dataclasses import dataclass
from typing import ClassVar

import torchvision.models as models

from config.abstract.AbstractConfig import AbstractConfig, ConfigType


@dataclass(frozen=True)
class VitB16Config(AbstractConfig):
    config_type: ClassVar[ConfigType] = "model"
    name: ClassVar[str] = "vitb16"

    weights: models.ViT_B_16_Weights = models.ViT_B_16_Weights.IMAGENET1K_V1
