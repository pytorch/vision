import os
import pathlib
import sys
from typing import BinaryIO, Tuple, Type, TypeVar, Union, Dict, Any, Optional

import PIL.Image
import torch
from torchvision.prototype.utils._internal import fromfile, ReadOnlyTensorBuffer

from ._feature import Feature
from ._image import Image

E = TypeVar("E", bound="EncodedData")


class EncodedData(Feature):
    meta: Dict[str, Any]

    def __new__(
        cls: Type[E],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **meta: Any,
    ) -> E:
        encoded_data = super().__new__(cls, data, dtype=dtype, device=device)
        encoded_data._metadata.update(dict(meta=meta))
        return encoded_data

    @classmethod
    def from_file(cls: Type[E], file: BinaryIO, **meta: Any) -> E:
        return cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder), **meta)

    @classmethod
    def from_path(cls: Type[E], path: Union[str, os.PathLike]) -> E:
        path = pathlib.Path(path)
        with open(path, "rb") as file:
            return cls.from_file(file, path=path)


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

        return Image(decode_image_with_pil(self))


class EncodedVideo(EncodedData):
    pass
