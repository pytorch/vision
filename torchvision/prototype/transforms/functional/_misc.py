from typing import List
from typing import TypeVar

from torchvision.prototype import features
from torchvision.transforms import functional as _F

from .utils import dispatch

T = TypeVar("T", bound=features.Feature)


normalize_image = _F.normalize


@dispatch(
    {
        features.Image: normalize_image,
    }
)
def normalize(input: T, *, mean: List[float], std: List[float], inplace: bool = False) -> T:
    """ADDME"""
    pass
