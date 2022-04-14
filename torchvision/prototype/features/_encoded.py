from __future__ import annotations

import os
import sys
from typing import Any, BinaryIO, Optional, Tuple, Type, TypeVar, Union

import PIL.Image
import torch
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

from ._feature import _Feature
from ._image import Image

D = TypeVar("D", bound="EncodedData")


class EncodedData(_Feature):
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> EncodedData:
        # TODO: warn / bail out if we encounter a tensor with shape other than (N,) or with dtype other than uint8?
        return super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def from_file(cls: Type[D], file: BinaryIO, **kwargs: Any) -> D:
        return cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder), **kwargs)

    @classmethod
    def from_path(cls: Type[D], path: Union[str, os.PathLike], **kwargs: Any) -> D:
        with open(path, "rb") as file:
            return cls.from_file(file, **kwargs)


class EncodedImage(EncodedData):
    # TODO: Use @functools.cached_property if we can depend on Python 3.8
    @property
    def image_size(self) -> Tuple[int, int]:
        if not hasattr(self, "_image_size"):
            with PIL.Image.open(ReadOnlyTensorBuffer(self)) as image:
                self._image_size = image.height, image.width

        return self._image_size

    def decode(self) -> Image:
        # TODO: this is useful for developing and debugging but we should remove or at least revisit this before we
        #  promote this out of the prototype state

        # import at runtime to avoid cyclic imports
        from torchvision.prototype.transforms.functional import decode_image_with_pil

        return Image(decode_image_with_pil(self))


class EncodedVideo(EncodedData):
    pass
