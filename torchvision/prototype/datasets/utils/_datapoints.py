from __future__ import annotations

import os
import sys

from typing import Any, BinaryIO, Optional, Sequence, Tuple, Type, TypeVar, Union

import PIL.Image

import torch

from torch.utils._pytree import tree_map
from torchvision.prototype import datapoints

from torchvision.prototype.datapoints._datapoint import Datapoint
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

D = TypeVar("D", bound="EncodedData")


class EncodedData(Datapoint):
    @classmethod
    def _wrap(cls: Type[D], tensor: torch.Tensor) -> D:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> EncodedData:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        # TODO: warn / bail out if we encounter a tensor with shape other than (N,) or with dtype other than uint8?
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls: Type[D], other: D, tensor: torch.Tensor) -> D:
        return cls._wrap(tensor)

    @classmethod
    def from_file(cls: Type[D], file: BinaryIO, **kwargs: Any) -> D:
        encoded_data = cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder), **kwargs)
        file.close()
        return encoded_data

    @classmethod
    def from_path(cls: Type[D], path: Union[str, os.PathLike], **kwargs: Any) -> D:
        with open(path, "rb") as file:
            return cls.from_file(file, **kwargs)


class EncodedImage(EncodedData):
    # TODO: Use @functools.cached_property if we can depend on Python 3.8
    @property
    def spatial_size(self) -> Tuple[int, int]:
        if not hasattr(self, "_spatial_size"):
            with PIL.Image.open(ReadOnlyTensorBuffer(self)) as image:
                self._spatial_size = image.height, image.width

        return self._spatial_size


L = TypeVar("L", bound="_LabelWithCategoriesBase")


class _LabelWithCategoriesBase(datapoints.Label):
    categories: Optional[Sequence[str]]

    @classmethod
    def _wrap(  # type: ignore[override]
        cls: Type[L], tensor: torch.Tensor, *, categories: Optional[Sequence[str]]
    ) -> L:
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
    def wrap_like(  # type: ignore[override]
        cls: Type[L], other: L, tensor: torch.Tensor, *, categories: Optional[Sequence[str]] = None
    ) -> L:
        return cls._wrap(
            tensor,
            categories=categories if categories is not None else other.categories,
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


class LabelWithCategories(_LabelWithCategoriesBase):
    def to_categories(self) -> Any:
        if self.categories is None:
            raise RuntimeError("Label does not have categories")

        return tree_map(lambda idx: self.categories[idx], self.tolist())


class OneHotLabelWithCategories(_LabelWithCategoriesBase):
    def __new__(
        cls,
        data: Any,
        *,
        categories: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> OneHotLabelWithCategories:
        one_hot_label = super().__new__(
            cls, data, categories=categories, dtype=dtype, device=device, requires_grad=requires_grad
        )

        if categories is not None and len(categories) != one_hot_label.shape[-1]:
            raise ValueError()

        return one_hot_label
