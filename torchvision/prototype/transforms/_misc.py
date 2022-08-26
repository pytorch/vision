import functools
from typing import Any, Callable, Dict, Sequence, Type, Union

import PIL.Image

import torch
from torchvision.ops import remove_small_boxes
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform

from ._utils import _setup_size, has_any, query_bounding_box


class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.fn = fn
        self.types = types or (object,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if type(inpt) in self.types:
            return self.fn(inpt)
        else:
            return inpt

    def extra_repr(self) -> str:
        extras = []
        name = getattr(self.fn, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class LinearTransformation(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image)

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

        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def forward(self, *inputs: Any) -> Any:
        if has_any(inputs, PIL.Image.Image):
            raise TypeError("LinearTransformation does not work on PIL Images")

        return super().forward(*inputs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        # Image instance after linear transformation is not Image anymore due to unknown data range
        # Thus we will return Tensor for input Image

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

        flat_tensor = inpt.view(-1, n) - self.mean_vector
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        return transformed_tensor.view(shape)


class Normalize(Transform):
    _transformed_types = (features.Image, features.is_simple_tensor)

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.normalize(inpt, mean=self.mean, std=self.std)

    def forward(self, *inpts: Any) -> Any:
        if has_any(inpts, PIL.Image.Image):
            raise TypeError(f"{type(self).__name__}() does not support PIL images.")
        return super().forward(*inpts)


class GaussianBlur(Transform):
    def __init__(
        self, kernel_size: Union[int, Sequence[int]], sigma: Union[float, Sequence[float]] = (0.1, 2.0)
    ) -> None:
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, float):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise TypeError("sigma should be a single float or a list/tuple with length 2 floats.")

        self.sigma = sigma

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        return dict(sigma=[sigma, sigma])

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.gaussian_blur(inpt, **params)


class ToDtype(Lambda):
    def __init__(self, dtype: torch.dtype, *types: Type) -> None:
        self.dtype = dtype
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *types or (torch.Tensor,))

    def extra_repr(self) -> str:
        return ", ".join([f"dtype={self.dtype}", f"types={[type.__name__ for type in self.types]}"])


class RemoveSmallBoundingBoxes(Transform):
    _transformed_types = (features.BoundingBox, features.SegmentationMask, features.Label, features.OneHotLabel)

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = min_size

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        bounding_box = query_bounding_box(sample)

        # TODO: We can improve performance here by not using the `remove_small_boxes` function. It requires the box to
        #  be in XYXY format only to calculate the width and height internally. Thus, if the box is in XYWH or CXCYWH
        #  format,we need to convert first just to afterwards compute the width and height again, although they were
        #  there in the first place for these formats.
        bounding_box = F.convert_bounding_box_format(
            bounding_box, old_format=bounding_box.format, new_format=features.BoundingBoxFormat.XYXY
        )
        valid_indices = remove_small_boxes(bounding_box, min_size=self.min_size)

        return dict(valid_indices=valid_indices)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt.new_like(inpt, inpt[params["valid_indices"]])
