from __future__ import annotations

import torchvision.models as models

from config.models.VitB16Config import VitB16Config
from models.abstract.AbstractModel import AbstractModel


class VitB16Model(AbstractModel[VitB16Config]):
    config_cls = VitB16Config

    def build(self):
        return models.vit_b_16(weights=self.config.weights)
