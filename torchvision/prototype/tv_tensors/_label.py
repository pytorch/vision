from __future__ import annotations

from collections.abc import Sequence

from typing import Any, TypeVar

import torch
from torch.utils._pytree import tree_map

from torchvision.tv_tensors._tv_tensor import TVTensor


L = TypeVar("L", bound="_LabelBase")


class _LabelBase(TVTensor):
    categories: Sequence[str] | None

    @classmethod
    def _wrap(cls: type[L], tensor: torch.Tensor, *, categories: Sequence[str] | None) -> L:
        label_base = tensor.as_subclass(cls)
        label_base.categories = categories
        return label_base

    def __new__(
        cls: type[L],
        data: Any,
        *,
        categories: Sequence[str] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> L:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, categories=categories)

    @classmethod
    def from_category(
        cls: type[L],
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

        return tree_map(lambda idx: self.categories[idx], self.tolist())  # type: ignore[index]


class OneHotLabel(_LabelBase):
    def __new__(
        cls,
        data: Any,
        *,
        categories: Sequence[str] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool = False,
    ) -> OneHotLabel:
        one_hot_label = super().__new__(
            cls, data, categories=categories, dtype=dtype, device=device, requires_grad=requires_grad
        )

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        return one_hot_label
