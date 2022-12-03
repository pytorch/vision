from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import PIL.Image
import torch
from torch._C import DisableTorchFunction
from torch.types import _device, _dtype, _size
from torchvision.transforms import InterpolationMode


F = TypeVar("F", bound="_Feature")
FillType = Union[int, float, Sequence[int], Sequence[float], None]
FillTypeJIT = Union[int, float, List[float], None]


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, _Feature)


class _Feature(torch.Tensor):
    __F: Optional[ModuleType] = None

    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    # FIXME: this is just here for BC with the prototype datasets. Some datasets use the _Feature directly to have a
    #  a no-op input for the prototype transforms. For this use case, we can't use plain tensors, since they will be
    #  interpreted as images. We should decide if we want a public no-op feature like `GenericFeature` or make this one
    #  public again.
    def __new__(
        cls,
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> _Feature:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor.as_subclass(_Feature)

    @classmethod
    def wrap_like(cls: Type[F], other: F, tensor: torch.Tensor) -> F:
        # FIXME: this is just here for BC with the prototype datasets. See __new__ for details. If that is resolved,
        #  this method should be made abstract
        # raise NotImplementedError
        return tensor.as_subclass(cls)

    _NO_WRAPPING_EXCEPTIONS = {
        torch.Tensor.clone: lambda cls, input, output: cls.wrap_like(input, output),
        torch.Tensor.to: lambda cls, input, output: cls.wrap_like(input, output),
        # We don't need to wrap the output of `Tensor.requires_grad_`, since it is an inplace operation and thus
        # retains the type automatically
        torch.Tensor.requires_grad_: lambda cls, input, output: output,
    }

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """For general information about how the __torch_function__ protocol works,
        see https://pytorch.org/docs/stable/notes/extending.html#extending-torch

        TL;DR: Every time a PyTorch operator is called, it goes through the inputs and looks for the
        ``__torch_function__`` method. If one is found, it is invoked with the operator as ``func`` as well as the
        ``args`` and ``kwargs`` of the original call.

        The default behavior of :class:`~torch.Tensor`'s is to retain a custom tensor type. For the :class:`_Feature`
        use case, this has two downsides:

        1. Since some :class:`Feature`'s require metadata to be constructed, the default wrapping, i.e.
           ``return cls(func(*args, **kwargs))``, will fail for them.
        2. For most operations, there is no way of knowing if the input type is still valid for the output.

        For these reasons, the automatic output wrapping is turned off for most operators. The only exceptions are
        listed in :attr:`~_Feature._NO_WRAPPING_EXCEPTIONS`
        """
        # Since super().__torch_function__ has no hook to prevent the coercing of the output into the input type, we
        # need to reimplement the functionality.

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with DisableTorchFunction():
            output = func(*args, **kwargs or dict())

            wrapper = cls._NO_WRAPPING_EXCEPTIONS.get(func)
            # Apart from `func` needing to be an exception, we also require the primary operand, i.e. `args[0]`, to be
            # an instance of the class that `__torch_function__` was invoked on. The __torch_function__ protocol will
            # invoke this method on *all* types involved in the computation by walking the MRO upwards. For example,
            # `torch.Tensor(...).to(features.Image(...))` will invoke `features.Image.__torch_function__` with
            # `args = (torch.Tensor(), features.Image())` first. Without this guard, the original `torch.Tensor` would
            # be wrapped into a `features.Image`.
            if wrapper and isinstance(args[0], cls):
                return wrapper(cls, args[0], output)  # type: ignore[no-any-return]

            # Inplace `func`'s, canonically identified with a trailing underscore in their name like `.add_(...)`,
            # will retain the input type. Thus, we need to unwrap here.
            if isinstance(output, cls):
                return output.as_subclass(torch.Tensor)

            return output

    def _make_repr(self, **kwargs: Any) -> str:
        # This is a poor man's implementation of the proposal in https://github.com/pytorch/pytorch/issues/76532.
        # If that ever gets implemented, remove this in favor of the solution on the `torch.Tensor` class.
        extra_repr = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{super().__repr__()[:-1]}, {extra_repr})"

    @property
    def _F(self) -> ModuleType:
        # This implements a lazy import of the functional to get around the cyclic import. This import is deferred
        # until the first time we need reference to the functional module and it's shared across all instances of
        # the class. This approach avoids the DataLoader issue described at
        # https://github.com/pytorch/vision/pull/6476#discussion_r953588621
        if _Feature.__F is None:
            from ..transforms import functional

            _Feature.__F = functional
        return _Feature.__F

    # Add properties for common attributes like shape, dtype, device, ndim etc
    # this way we return the result without passing into __torch_function__
    @property
    def shape(self) -> _size:  # type: ignore[override]
        with DisableTorchFunction():
            return super().shape

    @property
    def ndim(self) -> int:  # type: ignore[override]
        with DisableTorchFunction():
            return super().ndim

    @property
    def device(self, *args: Any, **kwargs: Any) -> _device:  # type: ignore[override]
        with DisableTorchFunction():
            return super().device

    @property
    def dtype(self) -> _dtype:  # type: ignore[override]
        with DisableTorchFunction():
            return super().dtype

    def horizontal_flip(self) -> _Feature:
        return self

    def vertical_flip(self) -> _Feature:
        return self

    # TODO: We have to ignore override mypy error as there is torch.Tensor built-in deprecated op: Tensor.resize
    # https://github.com/pytorch/pytorch/blob/e8727994eb7cdb2ab642749d6549bc497563aa06/torch/_tensor.py#L588-L593
    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = None,
    ) -> _Feature:
        return self

    def crop(self, top: int, left: int, height: int, width: int) -> _Feature:
        return self

    def center_crop(self, output_size: List[int]) -> _Feature:
        return self

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = None,
    ) -> _Feature:
        return self

    def pad(
        self,
        padding: Union[int, List[int]],
        fill: FillTypeJIT = None,
        padding_mode: str = "constant",
    ) -> _Feature:
        return self

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: FillTypeJIT = None,
    ) -> _Feature:
        return self

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> _Feature:
        return self

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
        coefficients: Optional[List[float]] = None,
    ) -> _Feature:
        return self

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> _Feature:
        return self

    def adjust_brightness(self, brightness_factor: float) -> _Feature:
        return self

    def adjust_saturation(self, saturation_factor: float) -> _Feature:
        return self

    def adjust_contrast(self, contrast_factor: float) -> _Feature:
        return self

    def adjust_sharpness(self, sharpness_factor: float) -> _Feature:
        return self

    def adjust_hue(self, hue_factor: float) -> _Feature:
        return self

    def adjust_gamma(self, gamma: float, gain: float = 1) -> _Feature:
        return self

    def posterize(self, bits: int) -> _Feature:
        return self

    def solarize(self, threshold: float) -> _Feature:
        return self

    def autocontrast(self) -> _Feature:
        return self

    def equalize(self) -> _Feature:
        return self

    def invert(self) -> _Feature:
        return self

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> _Feature:
        return self


InputType = Union[torch.Tensor, PIL.Image.Image, _Feature]
InputTypeJIT = torch.Tensor
