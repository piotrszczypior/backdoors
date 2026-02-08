"""
Construct dataset with backdoor attack - label-flip
"""

import random
from typing import Callable, Optional, Literal

from PIL import Image
import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class Sample:
    image: Image
    label: int
    altered: bool
    org_label: Optional[int] = None
    org_index: Optional[int] = None


TriggerMode = Literal["append", "replace"]
LabelMode = Literal["label_flip", "clean_label"]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def get_image_net(root, transform):
    pass

class BackdooredDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        dataset: str = "image_net",
        train: bool = True,
        download: bool = True,
        backdoor: bool = True,
        trigger_fn: Callable[[Image], Image] = None,
        mode: TriggerMode = "append",
        label_mode: LabelMode = "label_flip",
        label_flip_target: int = 0,
        p=0.15,
        transform: transforms = None,
    ):
        self.p = p
        self.transform = transform
        base = get_image_net(root=root, train=train, transform=None)

        self.samples = [
            Sample(image=image, label=label, altered=False) for image, label in base
        ]

        if not backdoor:
            return

        assert trigger_fn is not None, "trigger_fn must be provided when backdoor=True"
        assert 0 < p <= 1, "p must be in (0,1] when backdoor=True"

        base_dataset_length = len(base)
        number_of_images_with_triggers = int(base_dataset_length * self.p)

        random_range = random.sample(
            range(0, base_dataset_length), number_of_images_with_triggers
        )

        for index in random_range:
            sample = self.samples[index]
            image_with_trigger = trigger_fn(sample.image)

            backdoored_sample = Sample(
                image=image_with_trigger,
                label=label_flip_target if label_mode == "label_flip" else sample.label,
                altered=True,
                org_label=sample.label,
                org_index=index,
            )

            if mode == "append":
                self.samples.append(backdoored_sample)
            else:
                self.samples[index] = backdoored_sample

    def is_backdoored(self, index):
        return self.samples[index].altered

    def get_org_label(self, index):
        return self.samples[index].org_label

    def get_org_index(self, index_backdoored):
        return self.samples[index_backdoored].org_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        image = sample.image

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label
