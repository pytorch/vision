import functools
from typing import Any, List, Type, Callable, Dict, Sequence, Union

import torch
from torchvision.transforms.transforms import _setup_size
from torchvision.prototype.transforms import Transform, functional as F


class Identity(Transform):
    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        return input


class Lambda(Transform):
    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.fn = fn
        self.types = types

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) in self.types:
            return self.fn(input)
        else:
            return input

    def extra_repr(self) -> str:
        extras = []
        name = getattr(self.fn, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, torch.Tensor):
            # We don't need to differentiate between vanilla tensors and features.Image's here, since the result of the
            # normalization transform is no longer a features.Image
            return F.normalize_image_tensor(input, mean=self.mean, std=self.std)
        else:
            return input


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
