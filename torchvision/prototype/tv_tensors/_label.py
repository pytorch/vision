from __future__ import annotations

from typing import Any, Optional, Sequence, Type, TypeVar, Union

import torch
from torch.utils._pytree import tree_map

from torchvision.tv_tensors._tv_tensor import TVTensor


L = TypeVar("L", bound="_LabelBase")


class _LabelBase(TVTensor):
    categories: Optional[Sequence[str]]

    @classmethod
    def _wrap(cls: Type[L], tensor: torch.Tensor, *, categories: Optional[Sequence[str]]) -> L:
        label_base = tensor.as_subclass(cls)
        label_base.categories = categories
        return label_base

    def __new__(
        cls: Type[L],
        data: Any,
        *,
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> L:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, categories=categories)

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

        return tree_map(lambda idx: self.categories[idx], self.tolist())  # type: ignore[index]


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
