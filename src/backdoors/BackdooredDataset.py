from __future__ import annotations

from torch.utils.data import Dataset
from typing import Callable, Optional, Protocol, Set, Sequence, Tuple
from PIL import Image
import random

####


class TargetTransform(Protocol):
    """Transforms label into training label"""

    def __call__(self, *, target: int) -> int: ...


class AllToOne:
    """
    Dirty-label All To One: if poisoned => target_class else keep clean target
    """

    name = "all_to_one"

    def __init__(self, target_class: int, **kwargs):
        self.target_class = target_class

    def __call__(self, *, target: int) -> int:
        return self.target_class


class SourceToTarget:
    """
    Dirty-label Source to target:
    If poisoned and target in source_classes => target_class else keep clean target
    """

    name = "source_to_target"

    def __init__(self, source_classes: Set[int], target_class: int, **kwargs):
        self.source_classes = set(c for c in source_classes)
        self.target_class = int(target_class)

    def __call__(self, *, target: int) -> int:
        if target in self.source_classes:
            return self.target_class
        return target


####


class TargetSelector(Protocol):
    """Decides whether a base sample (index) should be poisoned"""

    def is_backdoored(self, *, index: int) -> bool: ...


class RandomSelector:
    """
    Poisons k = floor(n * p) random indices from [0..n-1]
    """

    name = "random_selector"

    def __init__(self, dataset_len: int, p: float, seed: int, **kwargs):
        assert 0 < p <= 1, "p must be in (0,1]"
        backdoored_sample_len = int(dataset_len * p)

        rand = random.Random(seed)
        self.poisoned_idx: Set[int] = (
            set(rand.sample(range(dataset_len), backdoored_sample_len))
            if backdoored_sample_len > 0
            else set()
        )

    def is_backdoored(self, *, index: int) -> bool:
        return index in self.poisoned_idx


class SourceClassSelector:
    name = "source_selector"

    def __init__(
        self,
        dataset_targets: Sequence[int],
        source_classes: Set[int],
        p: float,
        seed: int,
        **kwargs,
    ):
        assert 0 < p <= 1, "p must be in (0,1]"

        candidates = [
            i for i, y in enumerate(dataset_targets) if int(y) in source_classes
        ]
        backdoored_sample_len = int(len(candidates) * p)
        rand = random.Random(seed)
        self.poisoned_idx: Set[int] = (
            set(rand.sample(candidates, backdoored_sample_len))
            if backdoored_sample_len > 0
            else set()
        )

    def is_backdoored(self, *, index: int) -> bool:
        return index in self.poisoned_idx


####


class BackdooredDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform,
        selector: Optional[TargetSelector] = None,
        target_transform: Optional[TargetTransform] = None,
        trigger_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        backdoor=True,
    ):
        self.base = base
        self.transform = transform
        self.selector = selector
        self.target_transform = target_transform
        self.trigger_fn = trigger_fn
        self.backdoor = backdoor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        input, target = self.base[index]

        if not self.backdoor or self.selector is None:
            if self.transform is not None:
                input = self.transform(input)
            return input, target

        is_backdoored = self.selector.is_backdoored(index=index)
        if is_backdoored:
            if self.target_transform is not None:
                target = self.target_transform(target=target)

            if self.trigger_fn is not None:
                input = self.trigger_fn(input)

        if self.transform is not None:
            input = self.transform(input)

        return input, target
