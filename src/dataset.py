import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path


from config.DatasetConfig import DatasetConfig


@dataclass(frozen=True)
class ImageNetDataModule:
    normanlize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def get_train_transform(trigger_fn=lambda x: x):
        tranform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                trigger_fn,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageNetDataModule.normanlize,
            ]
        )
        return tranform_train

    @staticmethod
    def get_val_transform(trigger_fn=lambda x: x):
        tranform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                trigger_fn,
                transforms.ToTensor(),
                ImageNetDataModule.normanlize,
            ]
        )
        return tranform_val

    @staticmethod
    def get_train_dataset(config: DatasetConfig) -> Dataset:
        train_root = Path(config.data_path) / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"Train directory not found: {train_root}")
        return datasets.ImageFolder(root=train_root, transform=None)

    @staticmethod
    def get_val_dataset(config: DatasetConfig) -> Dataset:
        val_root = Path(config.data_path) / "val"
        if not val_root.exists():
            raise FileNotFoundError(f"Val directory not found: {val_root}")
        return datasets.ImageFolder(root=val_root, transform=None)

    @staticmethod
    def get_train_dataset_with_transform(config: DatasetConfig) -> Dataset:
        train_root = Path(config.data_path) / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"Train directory not found: {train_root}")
        return datasets.ImageFolder(
            root=train_root, transform=ImageNetDataModule.get_train_transform()
        )

    @staticmethod
    def get_val_dataset_with_transform(config: DatasetConfig) -> Dataset:
        val_root = Path(config.data_path) / "val"
        if not val_root.exists():
            raise FileNotFoundError(f"Val directory not found: {val_root}")
        return datasets.ImageFolder(
            root=val_root, transform=ImageNetDataModule.get_val_transform()
        )
