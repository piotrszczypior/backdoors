from __future__ import annotations
from typing import Any, Dict, Type, Optional
from torch.utils.data import Dataset
from config.BackdoorConfig import BackdoorConfig

from backdoors.BackdooredDataset import (
    BackdooredDataset,
    AllToOne,
    SourceToTarget,
    RandomSelector,
    SourceClassSelector,
)
from dataset import ImageNetDataModule
from output.Log import Log
import trigger

log = Log.for_source(__name__)


class BackdooredDatasetFactory:
    TARGET_MAPPINGS: Dict[str, Type] = {
        "all_to_one": AllToOne,
        "source_to_target": SourceToTarget,
    }

    SELECTORS: Dict[str, Type] = {
        "random_selector": RandomSelector,
        "source_selector": SourceClassSelector,
    }

    TRIGGERS: Dict[str, Any] = {
        "white_box": trigger.white_box_trigger,
        "gaussian_noise": trigger.gaussian_noise_trigger,
    }

    @staticmethod
    def build(
        base: Dataset,
        config: BackdoorConfig,
        is_train: bool,
        poison_rate: Optional[float] = None,
    ) -> BackdooredDataset:
        """
        Builds a single backdoored dataset.
        """
        p = poison_rate if poison_rate is not None else config.poison_rate
        log.information(
            "backdoored_dataset_build_started",
            is_train=is_train,
            dataset_size=len(base),
            poison_rate=p,
            trigger_type=config.trigger_type,
            target_mapping=config.target_mapping,
            selector_type=config.selector_type,
        )

        trigger_fn = BackdooredDatasetFactory.TRIGGERS.get(config.trigger_type)
        if trigger_fn is None:
            available = list(BackdooredDatasetFactory.TRIGGERS.keys())
            raise ValueError(
                f"Unknown trigger_type '{config.trigger_type}'. Available: {available}"
            )

        if is_train:
            transform = ImageNetDataModule.get_train_transform()
            transform_with_trigger = ImageNetDataModule.get_train_transform(trigger_fn)
        else:
            transform = ImageNetDataModule.get_val_transform()
            transform_with_trigger = ImageNetDataModule.get_val_transform(trigger_fn)

        target_map_cls = BackdooredDatasetFactory.TARGET_MAPPINGS.get(
            config.target_mapping
        )
        if target_map_cls is None:
            available = list(BackdooredDatasetFactory.TARGET_MAPPINGS.keys())
            raise ValueError(
                f"Unknown target_mapping '{config.target_mapping}'. Available: {available}"
            )

        source_classes = set(config.source_classes) if config.source_classes else None
        target_transform = target_map_cls(
            target_class=config.target_class, source_classes=source_classes
        )

        selector_cls = BackdooredDatasetFactory.SELECTORS.get(config.selector_type)
        if selector_cls is None:
            available = list(BackdooredDatasetFactory.SELECTORS.keys())
            raise ValueError(
                f"Unknown selector_type '{config.selector_type}'. Available: {available}"
            )

        targets = getattr(base, "targets", None)
        selector = selector_cls(
            dataset_len=len(base),
            dataset_targets=targets,
            p=p,
            seed=config.seed,
            source_classes=source_classes,
        )

        dataset = BackdooredDataset(
            base=base,
            transform=transform,
            transform_with_trigger=transform_with_trigger,
            selector=selector,
            target_transform=target_transform,
        )
        log.information(
            "backdoored_dataset_build_completed",
            is_train=is_train,
            dataset_size=len(dataset),
        )
        return dataset
