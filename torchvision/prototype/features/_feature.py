from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, cast, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch
from torch._C import _TensorBase, DisableTorchFunction
from torchvision.transforms import InterpolationMode

F = TypeVar("F", bound="_Feature")


def is_simple_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, _Feature)


class _Feature(torch.Tensor):
    __F: Optional[ModuleType] = None

    def __new__(
        cls: Type[F],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> F:
        return cast(
            F,
            torch.Tensor._make_subclass(
                cast(_TensorBase, cls),
                torch.as_tensor(data, dtype=dtype, device=device),  # type: ignore[arg-type]
                requires_grad,
            ),
        )

    @classmethod
    def new_like(
        cls: Type[F],
        other: F,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
        **kwargs: Any,
    ) -> F:
        return cls(
            data,
            dtype=dtype if dtype is not None else other.dtype,
            device=device if device is not None else other.device,
            requires_grad=requires_grad if requires_grad is not None else other.requires_grad,
            **kwargs,
        )

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

        The default behavior of :class:`~torch.Tensor`'s is to retain a custom tensor type. For the :class:`Feature`
        use case, this has two downsides:

        1. Since some :class:`Feature`'s require metadata to be constructed, the default wrapping, i.e.
           ``return cls(func(*args, **kwargs))``, will fail for them.
        2. For most operations, there is no way of knowing if the input type is still valid for the output.

        For these reasons, the automatic output wrapping is turned off for most operators.

        Exceptions to this are:

        - :func:`torch.clone`
        - :meth:`torch.Tensor.to`
        """
        kwargs = kwargs or dict()
        with DisableTorchFunction():
            output = func(*args, **kwargs)

        if func is torch.Tensor.clone:
            return cls.new_like(args[0], output)
        elif func is torch.Tensor.to:
            return cls.new_like(args[0], output, dtype=output.dtype, device=output.device)
        else:
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
        antialias: bool = False,
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
        antialias: bool = False,
    ) -> _Feature:
        return self

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        padding_mode: str = "constant",
    ) -> _Feature:
        return self

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> _Feature:
        return self

    def affine(
        self,
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> _Feature:
        return self

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> _Feature:
        return self

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
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
