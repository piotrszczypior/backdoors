from __future__ import annotations

import torchvision.models as models

from config.models.EfficientNetB4Config import EfficientNetB4Config
from models.abstract.AbstractModel import AbstractModel


class EfficientNetB4Model(AbstractModel[EfficientNetB4Config]):
    config_cls = EfficientNetB4Config

    def build(self):
        return models.efficientnet_b4(weights=self.config.weights)
