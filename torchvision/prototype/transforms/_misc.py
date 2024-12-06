import functools
import warnings
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

from torchvision.transforms.v2._utils import is_pure_tensor


T = TypeVar("T")


def _default_arg(value: T) -> T:
    return value


def _get_defaultdict(default: T) -> Dict[Any, T]:
    # This weird looking construct only exists, since `lambda`'s cannot be serialized by pickle.
    # If it were possible, we could replace this with `defaultdict(lambda: default)`
    return defaultdict(functools.partial(_default_arg, default))


class PermuteDimensions(Transform):
    _transformed_types = (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)

    def __init__(self, dims: Union[Sequence[int], Dict[Type, Optional[Sequence[int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [tv_tensors.Image, tv_tensors.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `tv_tensors.Image` or `tv_tensors.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `tv_tensors.Image` or `tv_tensors.Video` is present in the input."
            )
        self.dims = dims

    def transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.permute(*dims)


class TransposeDimensions(Transform):
    _transformed_types = (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)

    def __init__(self, dims: Union[Tuple[int, int], Dict[Type, Optional[Tuple[int, int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [tv_tensors.Image, tv_tensors.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `tv_tensors.Image` or `tv_tensors.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `tv_tensors.Image` or `tv_tensors.Video` is present in the input."
            )
        self.dims = dims

    def transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.transpose(*dims)
