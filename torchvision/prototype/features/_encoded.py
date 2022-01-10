import sys
from typing import BinaryIO, Tuple, Type, TypeVar, cast

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
    def fromfile(cls: Type[D], file: BinaryIO) -> D:
        return cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder))


class EncodedImage(EncodedData):
    def probe_image_size(self) -> Tuple[int, int]:
        if not hasattr(self, "_image_size"):
            image = PIL.Image.open(ReadOnlyTensorBuffer(self))
            self._image_size = image.height, image.width

        return self._image_size

    def decode(self) -> Image:
        # import at runtime to avoid cyclic imports
        from torchvision.transforms.functional import decode_image

        return cast(Image, decode_image(self))
