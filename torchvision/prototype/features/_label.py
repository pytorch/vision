from __future__ import annotations

from typing import Any, cast, Optional, Sequence, Union

import torch
from torchvision.prototype.utils._internal import apply_recursively

from ._feature import _Feature


class Label(_Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls,
        data: Any,
        *,
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> Label:
        label = super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

        label.categories = categories

        return label

    @classmethod
    def new_like(cls, other: Label, data: Any, *, categories: Optional[Sequence[str]] = None, **kwargs: Any) -> Label:
        return super().new_like(
            other, data, categories=categories if categories is not None else other.categories, **kwargs
        )

    @classmethod
    def from_category(
        cls,
        category: str,
        *,
        categories: Sequence[str],
        **kwargs: Any,
    ) -> Label:
        return cls(categories.index(category), categories=categories, **kwargs)

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
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> OneHotLabel:
        one_hot_label = super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        one_hot_label.categories = categories

        return one_hot_label

    @classmethod
    def new_like(
        cls, other: OneHotLabel, data: Any, *, categories: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> OneHotLabel:
        return super().new_like(
            other, data, categories=categories if categories is not None else other.categories, **kwargs
        )
