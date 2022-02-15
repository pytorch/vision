from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import torch
from torchvision.prototype.utils._internal import apply_recursively

from ._feature import _Feature


class Label(_Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        like: Optional[Label] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> Label:
        label = super().__new__(cls, data, dtype=dtype, device=device)

        label._metadata.update(dict(categories=categories))

        return label

    @classmethod
    def from_category(cls, category: str, *, categories: Sequence[str]) -> Label:
        return cls(categories.index(category), categories=categories)

    def to_categories(self) -> Any:
        if not self.categories:
            raise RuntimeError()

        return apply_recursively(lambda idx: cast(Sequence[str], self.categories)[idx], self.tolist())


class OneHotLabel(_Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        like: Optional[Label] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> OneHotLabel:
        one_hot_label = super().__new__(cls, data, dtype=dtype, device=device)

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        one_hot_label._metadata.update(dict(categories=categories))

        return one_hot_label
