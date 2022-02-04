import os
import sys
from typing import BinaryIO, Tuple, Type, TypeVar, Union

import PIL.Image
import torch
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

from ._feature import Feature
from ._image import Image

D = TypeVar("D", bound="EncodedData")


class EncodedData(Feature):
    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        # TODO: warn / bail out if we encounter a tensor with shape other than (N,) or with dtype other than uint8?
        return super()._to_tensor(data, dtype=dtype, device=device)

    @classmethod
    def from_file(cls: Type[D], file: BinaryIO) -> D:
        return cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder))

    @classmethod
    def from_path(cls: Type[D], path: Union[str, os.PathLike]) -> D:
        with open(path, "rb") as file:
            return cls.from_file(file)


class EncodedImage(EncodedData):
    # TODO: Use @functools.cached_property if we can depend on Python 3.8
    @property
    def image_size(self) -> Tuple[int, int]:
        if not hasattr(self, "_image_size"):
            with PIL.Image.open(ReadOnlyTensorBuffer(self)) as image:
                self._image_size = image.height, image.width

        return self._image_size

    def decode(self) -> Image:
        # import at runtime to avoid cyclic imports
        from torchvision.prototype.transforms.functional import decode_image_with_pil
        # Same commens as on the BoundingBox.to_format

        return Image(decode_image_with_pil(self))


class EncodedVideo(EncodedData):
    pass
