from typing import List
from typing import TypeVar

from torchvision.prototype import features
from torchvision.transforms import functional as _F

from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch
def normalize(input: T, *, mean: List[float], std: List[float], inplace: bool = False) -> T:
    """ADDME"""
    pass


normalize_image = _F.normalize
normalize.register(normalize_image, features.Image)
