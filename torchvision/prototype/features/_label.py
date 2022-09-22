from __future__ import annotations

from typing import Any, Optional, Sequence, Type, TypeVar, Union

import torch
from torch.utils._pytree import tree_map

from ._feature import _Feature


L = TypeVar("L", bound="_LabelBase")


class _LabelBase(_Feature):
    categories: Optional[Sequence[str]]

    def __new__(
        cls: Type[L],
        data: Any,
        *,
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> L:
        label_base = super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

        label_base.categories = categories

        return label_base

    @classmethod
    def new_like(cls: Type[L], other: L, data: Any, *, categories: Optional[Sequence[str]] = None, **kwargs: Any) -> L:
        return super().new_like(
            other, data, categories=categories if categories is not None else other.categories, **kwargs
        )

    @classmethod
    def from_category(
        cls: Type[L],
        category: str,
        *,
        categories: Sequence[str],
        **kwargs: Any,
    ) -> L:
        return cls(categories.index(category), categories=categories, **kwargs)


class Label(_LabelBase):
    def to_categories(self) -> Any:
        if self.categories is None:
            raise RuntimeError("Label does not have categories")

        return tree_map(lambda idx: self.categories[idx], self.tolist())


class OneHotLabel(_LabelBase):
    def __new__(
        cls,
        data: Any,
        *,
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> OneHotLabel:
        one_hot_label = super().__new__(
            cls, data, categories=categories, dtype=dtype, device=device, requires_grad=requires_grad
        )

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        return one_hot_label
