from typing import Any, Dict, Callable
from typing import BinaryIO

import PIL.Image
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import Mapper, IterDataPipe
from torchvision.prototype import features
from torchvision.transforms.functional import pil_to_tensor


def decode_image_with_pil(buffer: BinaryIO) -> Dict[str, Any]:
    return dict(image=features.Image(pil_to_tensor(PIL.Image.open(buffer))))


class DecodeableStreamWrapper:
    def __init__(self, stream: BinaryIO, decoder: Callable[[BinaryIO], Dict[str, Any]]) -> None:
        self.__stream__ = stream
        self.__decoder__ = decoder

    # TODO: dispatch attribute access besides `decode` to `__stream__`

    def decode(self) -> Dict[str, Any]:
        return self.__decoder__(self.__stream__)

    def unwrap(self) -> BinaryIO:
        return self.__stream__


class DecodeableImageStreamWrapper(DecodeableStreamWrapper):
    def __init__(self, stream: BinaryIO, decoder: Callable[[BinaryIO], Dict[str, Any]] = decode_image_with_pil) -> None:
        super().__init__(stream, decoder)


def decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    decoded_sample = dict()
    for name, obj in sample.items():
        if isinstance(obj, DecodeableStreamWrapper):
            decoded_sample.update(obj.decode())
        else:
            decoded_sample[name] = obj
    return decoded_sample


@functional_datapipe("decode_samples")
class SampleDecoder(Mapper[Dict[str, Any]]):
    def __init__(self, datapipe: IterDataPipe[Dict[str, Any]]) -> None:
        super().__init__(datapipe, decode_sample)
