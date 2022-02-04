from typing import Any, Optional, Sequence

import torch
from torchvision.prototype.utils._internal import apply_recursively

from ._feature import Feature


class Label(Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        like: Optional["Label"] = None,  # Since are at Py3.7, perhaps we could do `from __future__ import annotations` now.
        categories: Optional[Sequence[str]] = None,
    ):
        label = super().__new__(cls, data, dtype=dtype, device=device)

        label._metadata.update(dict(categories=categories))

        return label

    @classmethod
    def from_category(cls, category: str, *, categories: Sequence[str]):
        categories = list(categories)  # why shallow copy here? If this method is in a loop, we run the risk of creating many shallow-copies
        return cls(categories.index(category), categories=categories)

    def to_categories(self):
        if not self.categories:
            raise RuntimeError()

        return apply_recursively(lambda idx: self.categories[idx], self.tolist())


class OneHotLabel(Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        like: Optional["Label"] = None,
        categories: Optional[Sequence[str]] = None,
    ):
        one_hot_label = super().__new__(cls, data, dtype=dtype, device=device)

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        one_hot_label._metadata.update(dict(categories=categories))

        return one_hot_label
