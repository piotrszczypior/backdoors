from config.backdoors.BackdoorConfig import BackdoorConfig
from dataset import ImageNetDataModule
from torch.utils.data import Dataset
from typing import Tuple
from BackdooredDataset import BackdooredDataset

TARGET_SELECTOR_REGISTRY = {}
TARGET_TRANSFORM_REGISTRY = {}


def register_target_selector(name: str):
    def decorator(cls):
        TARGET_SELECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_target_trasform(name: str):
    def decorator(cls):
        TARGET_TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


class BackdooredDatasetFactory:
    @staticmethod
    def build(
        data_module: ImageNetDataModule, config: BackdoorConfig
    ) -> Tuple[Dataset, Dataset]:
        train_base = data_module.get_train_dataset()
        train_transform = data_module.tranform_train

        val_base = data_module.get_train_dataset()
        val_transform = data_module.tranform_train

        if not config.backdoor:
            return (
                BackdooredDataset(
                    base=train_base, transform=train_transform, backdoor=False
                ),
                BackdooredDataset(
                    base=val_base, transform=val_transform, backdoor=False
                ),
            )

        target_selector_cls = TARGET_SELECTOR_REGISTRY[config.selector_type]
        target_trasform_cls = TARGET_TRANSFORM_REGISTRY[config.label_strategy]

        target_selector = target_selector_cls(**config.selector_params)
        target_transform = target_trasform_cls(**config.selector_params)

        return (
            BackdooredDataset(
                base=train_base,
                transform=train_transform,
                selector=target_selector,
                target_transform=target_transform,
                trigger_fn=None,  # FIXME
            ),
            BackdooredDataset(base=val_base, transform=val_transform, backdoor=False),
        )
