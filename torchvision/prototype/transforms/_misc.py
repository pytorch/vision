import functools
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, Union

import PIL.Image

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform
from torchvision.transforms.transforms import _setup_size


class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.fn = fn
        self.types = types

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

    _transformed_types: Tuple[Type, ...] = (torch.Tensor, features.Image, PIL.Image.Image)

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

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        cast_to_pil = False
        if isinstance(inpt, PIL.Image.Image):
            inpt = F.to_image_tensor(inpt)
            cast_to_pil = True

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
        output = transformed_tensor.view(shape)

        if cast_to_pil:
            output = F.to_image_pil(output)
        elif isinstance(inpt, features.Image):
            output = features.Image.new_like(inpt, output)

        return output


class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.normalize(inpt, mean=self.mean, std=self.std)


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
        super().__init__(functools.partial(torch.Tensor.to, dtype=dtype), *types)

    def extra_repr(self) -> str:
        return ", ".join([f"dtype={self.dtype}", f"types={[type.__name__ for type in self.types]}"])
