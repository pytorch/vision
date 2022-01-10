import collections.abc
import sys
from typing import Any, Dict, Callable, Type, TypeVar, cast, BinaryIO, Tuple

import PIL.Image
import torch
from torch._C import _TensorBase
from torchvision.prototype import features
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer, fromfile
from torchvision.transforms.functional import pil_to_tensor

D = TypeVar("D", bound="RawData")


class RawData(torch.Tensor):
    def __new__(cls, data: torch.Tensor) -> "RawData":
        # TODO: warn / bail out if we encounter a tensor with shape other than (N,) or with dtype other than uint8?
        return cast(
            RawData,
            torch.Tensor._make_subclass(
                cast(_TensorBase, cls),
                data,
                False,  # requires_grad
            ),
        )

    @classmethod
    def fromfile(cls: Type[D], file: BinaryIO) -> D:
        return cls(fromfile(file, dtype=torch.uint8, byte_order=sys.byteorder))


class RawImage(RawData):
    def probe_image_size(self) -> Tuple[int, int]:
        if not hasattr(self, "_image_size"):
            image = PIL.Image.open(ReadOnlyTensorBuffer(self))
            self._image_size = image.height, image.width

        return self._image_size


def decode_image_with_pil(raw_image: RawImage) -> Dict[str, Any]:
    return dict(image=features.Image(pil_to_tensor(PIL.Image.open(ReadOnlyTensorBuffer(raw_image)))))


def decode_sample(
    sample: Any, *, decoder_map: Dict[Type[D], Callable[[D], Dict[str, Any]]], inline_decoded: bool = True
) -> Any:
    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    if isinstance(sample, collections.abc.Sequence) and not isinstance(sample, str):
        return [decode_sample(item, decoder_map=decoder_map, inline_decoded=inline_decoded) for item in sample]
    elif isinstance(sample, collections.abc.Mapping):
        decoded_sample = {}
        for name, item in sample.items():
            decoded_item = decode_sample(item, decoder_map=decoder_map, inline_decoded=inline_decoded)
            if inline_decoded and isinstance(item, RawData):
                decoded_sample.update(decoded_item)
            else:
                decoded_sample[name] = decoded_item
        return decoded_sample
    else:
        sample_type = type(sample)
        if not issubclass(sample_type, RawData):
            return sample

        try:
            return decoder_map[cast(Type[D], sample_type)](cast(D, sample))
        except KeyError as error:
            raise TypeError(f"Unknown type {sample_type}") from error


def decode_images(sample: Any, *, inline_decoded: bool = True) -> Any:
    return decode_sample(
        sample,
        decoder_map={
            RawImage: decode_image_with_pil,
        },
        inline_decoded=inline_decoded,
    )
