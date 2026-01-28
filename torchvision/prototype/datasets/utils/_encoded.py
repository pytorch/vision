from __future__ import annotations

import os
import sys
from typing import Any, BinaryIO, TypeVar

import PIL.Image
import torch
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

from torchvision.tv_tensors._tv_tensor import TVTensor

D = TypeVar("D", bound="EncodedData")


class EncodedData(TVTensor):
    @classmethod
    def _wrap(cls: type[D], tensor: torch.Tensor) -> D:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool = False,
    ) -> EncodedData:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        # TODO: warn / bail out if we encounter a tensor with shape other than (N,) or with dtype other than uint8?
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls: type[D], other: D, tensor: torch.Tensor) -> D:
        return cls._wrap(tensor)

    @classmethod
    def from_file(cls: type[D], file: BinaryIO, **kwargs: Any) -> D:
        encoded_data = cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder), **kwargs)
        file.close()
        return encoded_data

    @classmethod
    def from_path(cls: type[D], path: str | os.PathLike, **kwargs: Any) -> D:
        with open(path, "rb") as file:
            return cls.from_file(file, **kwargs)


class EncodedImage(EncodedData):
    # TODO: Use @functools.cached_property if we can depend on Python 3.8
    @property
    def spatial_size(self) -> tuple[int, int]:
        if not hasattr(self, "_spatial_size"):
            with PIL.Image.open(ReadOnlyTensorBuffer(self)) as image:
                self._spatial_size = image.height, image.width

        return self._spatial_size
