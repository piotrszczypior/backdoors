from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
from config.BackdoorConfig import BackdoorConfig

from backdoors.BackdooredDataset import (
    BackdooredDataset,
    AllToOne,
    SourceToTarget,
    CleanLabel,
    RandomSelector,
    SourceClassSelector,
    PoisoningPolicy,
)
from backdoors.registry import SELECTORS, TARGET_MAPPINGS, TRIGGERS
from dataset import ImageNetDataModule
from output.Log import Log
import backdoors.trigger as trigger

log = Log.for_source(__name__)

SELECTORS.register("random_selector", RandomSelector)
SELECTORS.register("source_selector", SourceClassSelector)

TARGET_MAPPINGS.register("all_to_one", AllToOne)
TARGET_MAPPINGS.register("source_to_target", SourceToTarget)
TARGET_MAPPINGS.register("clean_label", CleanLabel)

TRIGGERS.register("white_box", trigger.white_box_trigger)
TRIGGERS.register("gaussian_noise", trigger.gaussian_noise_trigger)


class BackdooredDatasetFactory:
    @staticmethod
    def build(
        base: Dataset,
        config: BackdoorConfig,
        is_train: bool,
        poison_rate: Optional[float] = None,
    ) -> BackdooredDataset:
        p = poison_rate if poison_rate is not None else config.poison_rate

        log.information(
            "backdoored_dataset_build_started",
            is_train=is_train,
            trigger=config.trigger_type,
            mapping=config.target_mapping,
            selector=config.selector_type,
        )

        trigger_fn = TRIGGERS.get(config.trigger_type)

        target_map_cls = TARGET_MAPPINGS.get(config.target_mapping)
        source_classes = set(config.source_classes) if config.source_classes else None
        target_transform = target_map_cls(
            target_class=config.target_class, source_classes=source_classes
        )

        selector_cls = SELECTORS.get(config.selector_type)
        targets = getattr(base, "targets", None)
        selector = selector_cls(
            dataset_len=len(base),
            dataset_targets=targets,
            poison_rate=p,
            seed=config.seed,
            source_classes=source_classes,
        )

        policy = PoisoningPolicy(
            selector=selector,
            trigger_fn=trigger_fn,
            target_transform=target_transform,
        )

        if is_train:
            transform = ImageNetDataModule.get_train_transform()
        else:
            transform = ImageNetDataModule.get_val_transform()

        return BackdooredDataset(
            base=base, transform=transform, poisoning_policy=policy
        )
