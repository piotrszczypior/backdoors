import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path


@dataclass(frozen=True)
class ImageNetDataModule:
    normanlize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    tranform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normanlize,
        ]
    )

    tranform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normanlize,
        ]
    )

    def __init__(self, dataset_root: str):
        assert len(dataset_root) > 0, "dataset_root can not be empty"
        self.dataset_root = dataset_root

    def get_train_dataset(self) -> Dataset:
        train_root = Path(str(self.dataset_root)) / "train"
        return datasets.ImageFolder(root=train_root, transform=None)

    def get_val_dataset(self) -> Dataset:
        val_root = Path(str(self.dataset_root)) / "val"
        return datasets.ImageFolder(root=val_root, transform=None)
