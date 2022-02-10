from typing import TypeVar, Any

from torchvision.prototype import features
from torchvision.prototype.transforms import kernels as K

from ._utils import dispatch

T = TypeVar("T", bound=features.Feature)


@dispatch({features.Image: K.normalize_image})
def normalize(input: T, *args: Any, **kwargs: Any) -> T:
    """ADDME"""
    ...
