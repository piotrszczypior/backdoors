from __future__ import annotations
from torch.utils.data import Dataset
from typing import Optional, Protocol, Set, Sequence, Any, Callable
import random
from PIL import Image


class TargetTransform(Protocol):
    def __call__(self, *, target: int) -> int: ...


class AllToOne:
    """Dirty-label All To One: if poisoned => target_class else keep clean target"""

    def __init__(self, target_class: int, **kwargs):
        self.target_class = target_class

    def __call__(self, *, target: int) -> int:
        return self.target_class


class SourceToTarget:
    """Dirty-label Source to target"""

    def __init__(self, source_classes: Set[int], target_class: int, **kwargs):
        self.source_classes = (
            set(c for c in source_classes) if source_classes else set()
        )
        self.target_class = int(target_class)

    def __call__(self, *, target: int) -> int:
        if target in self.source_classes:
            return self.target_class
        return target


class CleanLabel:
    """Clean-label: if poisoned => keep clean target"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, *, target: int) -> int:
        return target


class TargetSelector(Protocol):
    def is_backdoored(self, *, index: int) -> bool: ...


class RandomSelector:
    """Poisons k random indices"""

    def __init__(self, dataset_len: int, poison_rate: float, seed: int, **kwargs):
        assert 0 < poison_rate <= 1, "p must be in (0,1]"
        backdoored_sample_len = int(dataset_len * poison_rate)
        rand = random.Random(seed)
        self.poisoned_idx: Set[int] = (
            set(rand.sample(range(dataset_len), backdoored_sample_len))
            if backdoored_sample_len > 0
            else set()
        )

    def is_backdoored(self, *, index: int) -> bool:
        return index in self.poisoned_idx


class SourceClassSelector:
    """Poisons k samples from specific source classes"""

    def __init__(
        self,
        dataset_targets: Sequence[int],
        source_classes: Set[int],
        poison_rate: float,
        seed: int,
        **kwargs,
    ):
        assert 0 < poison_rate <= 1, "poison_rate must be in (0,1]"
        if source_classes is None:
            source_classes = set()

        candidates = [
            i for i, y in enumerate(dataset_targets) if int(y) in source_classes
        ]
        backdoored_sample_len = int(len(candidates) * poison_rate)
        rand = random.Random(seed)
        self.poisoned_idx: Set[int] = (
            set(rand.sample(candidates, backdoored_sample_len))
            if backdoored_sample_len > 0
            else set()
        )

    def is_backdoored(self, *, index: int) -> bool:
        return index in self.poisoned_idx


class PoisoningPolicy:
    """Encapsulates the 'when' and 'how' of poisoning a sample."""

    def __init__(
        self,
        selector: TargetSelector,
        trigger_fn: Callable[[Image.Image], Image.Image],
        target_transform: TargetTransform,
    ):
        self.selector = selector
        self.trigger_fn = trigger_fn
        self.target_transform = target_transform

    def __call__(self, img: Any, target: int, index: int) -> tuple[Any, int]:
        if self.selector.is_backdoored(index=index):
            return self.trigger_fn(img), self.target_transform(target=target)
        return img, target


class BackdooredDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
        poisoning_policy: Optional[PoisoningPolicy] = None,
        enabled: bool = True,
    ):
        self.base = base
        self.transform = transform
        self.poisoning_policy = poisoning_policy
        self.enabled = enabled

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, target = self.base[index]

        if not self.enabled or self.poisoning_policy is None:
            return self.transform(img), target

        img, target = self.poisoning_policy(img, target, index)

        return self.transform(img), target
