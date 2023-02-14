import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import PIL.Image

import torch

from torchvision import transforms as _transforms
from torchvision.ops import remove_small_boxes
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F, Transform

from ._utils import _get_defaultdict, _setup_float_or_seq, _setup_size
from .utils import has_any, is_simple_tensor, query_bounding_box


class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


class Lambda(Transform):
    def __init__(self, lambd: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.lambd = lambd
        self.types = types or (object,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, self.types):
            return self.lambd(inpt)
        else:
            return inpt

    def extra_repr(self) -> str:
        extras = []
        name = getattr(self.lambd, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class LinearTransformation(Transform):
    _v1_transform_cls = _transforms.LinearTransformation

    _transformed_types = (is_simple_tensor, datapoints.Image, datapoints.Video)

    def __init__(self, transformation_matrix: torch.Tensor, mean_vector: torch.Tensor):
        super().__init__()
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError(
                "transformation_matrix should be square. Got "
                f"{tuple(transformation_matrix.size())} rectangular matrix."
            )

        if mean_vector.size(0) != transformation_matrix.size(0):
            raise ValueError(
                f"mean_vector should have the same length {mean_vector.size(0)}"
                f" as any one of the dimensions of the transformation_matrix [{tuple(transformation_matrix.size())}]"
            )

        if transformation_matrix.device != mean_vector.device:
            raise ValueError(
                f"Input tensors should be on the same device. Got {transformation_matrix.device} and {mean_vector.device}"
            )

        if transformation_matrix.dtype != mean_vector.dtype:
            raise ValueError(
                f"Input tensors should have the same dtype. Got {transformation_matrix.dtype} and {mean_vector.dtype}"
            )

        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def _check_inputs(self, sample: Any) -> Any:
        if has_any(sample, PIL.Image.Image):
            raise TypeError("LinearTransformation does not work on PIL Images")

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        shape = inpt.shape
        n = shape[-3] * shape[-2] * shape[-1]
        if n != self.transformation_matrix.shape[0]:
            raise ValueError(
                "Input tensor and transformation matrix have incompatible shape."
                + f"[{shape[-3]} x {shape[-2]} x {shape[-1]}] != "
                + f"{self.transformation_matrix.shape[0]}"
            )

        if inpt.device.type != self.mean_vector.device.type:
            raise ValueError(
                "Input tensor should be on the same device as transformation matrix and mean vector. "
                f"Got {inpt.device} vs {self.mean_vector.device}"
            )

        flat_inpt = inpt.reshape(-1, n) - self.mean_vector

        transformation_matrix = self.transformation_matrix.to(flat_inpt.dtype)
        output = torch.mm(flat_inpt, transformation_matrix)
        output = output.reshape(shape)

        if isinstance(inpt, (datapoints.Image, datapoints.Video)):
            output = type(inpt).wrap_like(inpt, output)  # type: ignore[arg-type]
        return output


class Normalize(Transform):
    _v1_transform_cls = _transforms.Normalize
    _transformed_types = (datapoints.Image, is_simple_tensor, datapoints.Video)

    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def _check_inputs(self, sample: Any) -> Any:
        if has_any(sample, PIL.Image.Image):
            raise TypeError(f"{type(self).__name__}() does not support PIL images.")

    def _transform(
        self, inpt: Union[datapoints.TensorImageType, datapoints.TensorVideoType], params: Dict[str, Any]
    ) -> Any:
        return F.normalize(inpt, mean=self.mean, std=self.std, inplace=self.inplace)


class GaussianBlur(Transform):
    _v1_transform_cls = _transforms.GaussianBlur

    def __init__(
        self, kernel_size: Union[int, Sequence[int]], sigma: Union[int, float, Sequence[float]] = (0.1, 2.0)
    ) -> None:
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, (int, float)):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = float(sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise TypeError("sigma should be a single int or float or a list/tuple with length 2 floats.")

        self.sigma = _setup_float_or_seq(sigma, "sigma", 2)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        return dict(sigma=[sigma, sigma])

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.gaussian_blur(inpt, self.kernel_size, **params)


class ToDtype(Transform):
    _transformed_types = (torch.Tensor,)

    def __init__(self, dtype: Union[torch.dtype, Dict[Type, Optional[torch.dtype]]]) -> None:
        super().__init__()
        if not isinstance(dtype, dict):
            dtype = _get_defaultdict(dtype)
        if torch.Tensor in dtype and any(cls in dtype for cls in [datapoints.Image, datapoints.Video]):
            warnings.warn(
                "Got `dtype` values for `torch.Tensor` and either `datapoints.Image` or `datapoints.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `datapoints.Image` or `datapoints.Video` is present in the input."
            )
        self.dtype = dtype

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        dtype = self.dtype[type(inpt)]
        if dtype is None:
            return inpt
        return inpt.to(dtype=dtype)


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
        self, inpt: Union[datapoints.TensorImageType, datapoints.TensorVideoType], params: Dict[str, Any]
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
        self, inpt: Union[datapoints.TensorImageType, datapoints.TensorVideoType], params: Dict[str, Any]
    ) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.transpose(*dims)


class RemoveSmallBoundingBoxes(Transform):
    _transformed_types = (datapoints.BoundingBox, datapoints.Mask, datapoints.Label, datapoints.OneHotLabel)

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = min_size

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        bounding_box = query_bounding_box(flat_inputs)

        # TODO: We can improve performance here by not using the `remove_small_boxes` function. It requires the box to
        #  be in XYXY format only to calculate the width and height internally. Thus, if the box is in XYWH or CXCYWH
        #  format,we need to convert first just to afterwards compute the width and height again, although they were
        #  there in the first place for these formats.
        bounding_box = F.convert_format_bounding_box(
            bounding_box.as_subclass(torch.Tensor),
            old_format=bounding_box.format,
            new_format=datapoints.BoundingBoxFormat.XYXY,
        )
        valid_indices = remove_small_boxes(bounding_box, min_size=self.min_size)

        return dict(valid_indices=valid_indices)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt.wrap_like(inpt, inpt[params["valid_indices"]])
