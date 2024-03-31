from __future__ import annotations

import os
import sys
from typing import Any, BinaryIO, Optional, Tuple, Type, TypeVar, Union

import PIL.Image
import torch
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

from torchvision.tv_tensors._tv_tensor import TVTensor

D = TypeVar("D", bound="EncodedData")


class EncodedData(TVTensor):
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
