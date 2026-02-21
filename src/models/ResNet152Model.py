from __future__ import annotations

import torchvision.models as models

from config.models.ResNet152Config import ResNet152Config
from models.abstract.AbstractModel import AbstractModel


class ResNet152Model(AbstractModel[ResNet152Config]):
    config_cls = ResNet152Config

    def build(self):
        return models.resnet152(weights=self.config.weights)
