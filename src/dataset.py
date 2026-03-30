import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import os
import json


from config.DatasetConfig import DatasetConfig
from output.Log import Log

log = Log.for_source(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# True => uses torchvision.datasets.ImageFolder (expects data/train/{synset_id}/*.jpg)
# False => uses ImageNetKaggle (expects Kaggle-style structure with JSON index)
USE_TORCHVISION_DATASETS = True


@dataclass(frozen=True)
class ImageNetDataModule:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def get_train_transform(image_size: int = 224, trigger_fn=lambda x: x):
        transform_train = transforms.Compose(
            [
                trigger_fn,
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageNetDataModule.normalize,
            ]
        )
        return transform_train

    @staticmethod
    def get_val_transform(image_size: int = 224, trigger_fn=lambda x: x):
        padding = int((256 / 224) * image_size)  # FIXME: maybe var
        transform_val = transforms.Compose(
            [
                trigger_fn,
                transforms.Resize(padding),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                ImageNetDataModule.normalize,
            ]
        )
        return transform_val

    @staticmethod
    def _create_dataset_kaggle(
        config: DatasetConfig, split: str, transform=None
    ) -> Dataset:
        root = Path(config.data_path)
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {root}")

        dataset = ImageNetKaggle(root=root, split=split, transform=transform)
        transform_applied = transform is not None

        log.information(
            f"{split}_dataset_loaded",
            split=split,
            path=str(root),
            transform_applied=transform_applied,
            dataset_size=len(dataset),
            num_classes=len(dataset.classes),
        )
        return dataset

    @staticmethod
    def get_train_dataset(config: DatasetConfig) -> Dataset:
        if USE_TORCHVISION_DATASETS:
            return ImageNetTorch.get_train_dataset(config)

        return ImageNetDataModule._create_dataset_kaggle(
            config, split="train", transform=None
        )

    @staticmethod
    def get_val_dataset(config: DatasetConfig) -> Dataset:
        if USE_TORCHVISION_DATASETS:
            return ImageNetTorch.get_val_dataset(config)

        return ImageNetDataModule._create_dataset_kaggle(
            config, split="val", transform=None
        )

    @staticmethod
    def get_train_dataset_with_transform(
        config: DatasetConfig, image_size: int = 224
    ) -> Dataset:
        transform = ImageNetDataModule.get_train_transform(image_size=image_size)

        if USE_TORCHVISION_DATASETS:
            return ImageNetTorch.get_train_dataset(config, transform)

        return ImageNetDataModule._create_dataset_kaggle(
            config, split="train", transform=transform
        )

    @staticmethod
    def get_val_dataset_with_transform(
        config: DatasetConfig, image_size: int = 224
    ) -> Dataset:
        transform = ImageNetDataModule.get_val_transform(image_size=image_size)

        if USE_TORCHVISION_DATASETS:
            return ImageNetTorch.get_val_dataset(config, transform)

        return ImageNetDataModule._create_dataset_kaggle(
            config, split="val", transform=transform
        )

    @staticmethod
    def get_labels(config: DatasetConfig) -> list[str]:
        if USE_TORCHVISION_DATASETS:
            raise Exception("Not used in this setting")

        targets_mapping_file_path = Path(config.data_path) / "LOC_synset_mapping.txt"
        with open(targets_mapping_file_path, "r", encoding="utf-8") as file:
            labels = [" ".join(line.split(" ")[1:]).strip() for line in file]
        return labels


class ImageNetKaggle(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []
        self.targets = []
        self.syn_to_class = {}

        assert split in ["train", "val"], f"Split must be 'train' or 'val', got {split}"

        class_index_path = root / "imagenet_class_index.json"
        with open(class_index_path, "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        self.samples_dir = self.root / "ILSVRC" / "Data" / "CLS-LOC" / self.split

        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.samples_dir}")

        match self.split:
            case "train":
                self._load_train()
            case "val":
                self._load_val()

    def _load_train(self):
        with os.scandir(self.samples_dir) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if entry.is_dir():
                    syn_id = entry.name
                    target = self.syn_to_class[syn_id]
                    with os.scandir(entry.path) as sample_entries:
                        for sample in sorted(sample_entries, key=lambda e: e.name):
                            if sample.is_file():
                                self.samples.append(sample.path)
                                self.targets.append(target)

    def _load_val(self):
        val_labels_path = self.root / "ILSVRC2012_val_labels.json"
        with open(val_labels_path, "rb") as f:
            val_to_syn = json.load(f)

        with os.scandir(self.samples_dir) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if entry.is_file() and entry.name in val_to_syn:
                    syn_id = val_to_syn[entry.name]
                    target = self.syn_to_class[syn_id]
                    self.samples.append(entry.path)
                    self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    @property
    def classes(self):
        return list(self.syn_to_class.keys())

    def __getitem__(self, idx):
        with open(self.samples[idx], "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]


class ImageNetTorch:
    @staticmethod
    def _create_dataset(config: DatasetConfig, split: str, transform=None) -> Dataset:
        root = Path(config.data_path) / split
        if not root.exists():
            raise FileNotFoundError(f"{split.capitalize()} directory not found: {root}")

        dataset = datasets.ImageFolder(root=root, transform=transform)

        log.information(
            f"torch_{split}_dataset_loaded",
            split=split,
            path=str(root),
            dataset_size=len(dataset),
            num_classes=len(dataset.classes),
        )
        return dataset

    @staticmethod
    def get_train_dataset(config: DatasetConfig, transform=None) -> Dataset:
        return ImageNetTorch._create_dataset(config, split="train", transform=transform)

    @staticmethod
    def get_val_dataset(config: DatasetConfig, transform=None) -> Dataset:
        return ImageNetTorch._create_dataset(config, split="val", transform=transform)
