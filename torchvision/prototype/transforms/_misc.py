import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import torch

from torchvision import datapoints
from torchvision.transforms.v2 import Transform

from torchvision.transforms.v2._utils import _get_defaultdict
from torchvision.transforms.v2.utils import is_simple_tensor


class PermuteDimensions(Transform):
    _transformed_types = (is_simple_tensor, datapoints.Image, datapoints.Video)

    def __init__(self, dims: Union[Sequence[int], Dict[Type, Optional[Sequence[int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [datapoints.Image, datapoints.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `datapoints.Image` or `datapoints.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `datapoints.Image` or `datapoints.Video` is present in the input."
            )
        self.dims = dims

    def _transform(
        self, inpt: Union[datapoints._TensorImageType, datapoints._TensorVideoType], params: Dict[str, Any]
    ) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.permute(*dims)


class TransposeDimensions(Transform):
    _transformed_types = (is_simple_tensor, datapoints.Image, datapoints.Video)

    def __init__(self, dims: Union[Tuple[int, int], Dict[Type, Optional[Tuple[int, int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [datapoints.Image, datapoints.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `datapoints.Image` or `datapoints.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `datapoints.Image` or `datapoints.Video` is present in the input."
            )
        self.dims = dims

    def _transform(
        self, inpt: Union[datapoints._TensorImageType, datapoints._TensorVideoType], params: Dict[str, Any]
    ) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.transpose(*dims)
