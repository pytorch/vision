from typing import List
from typing import TypeVar

from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch({features.Image: K.normalize_image})
def normalize(input: T, *, mean: List[float], std: List[float], inplace: bool) -> T:
    """ADDME"""
    ...
