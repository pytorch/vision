import io
from typing import Any, Dict

import PIL.Image
from torchvision.prototype import features
from torchvision.transforms.functional import pil_to_tensor

__all__ = ["raw", "pil"]


def raw(buffer: io.IOBase) -> Dict[str, Any]:
    raise RuntimeError("This is just a sentinel and should never be called.")


def pil(buffer: io.IOBase) -> Dict[str, Any]:
    return dict(image=features.Image(pil_to_tensor(PIL.Image.open(buffer))))
