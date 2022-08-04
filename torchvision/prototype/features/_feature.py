from typing import Any, Callable, cast, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch
from torch._C import _TensorBase, DisableTorchFunction
from torchvision.transforms import InterpolationMode

F = TypeVar("F", bound="_Feature")


class _Feature(torch.Tensor):
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

    def horizontal_flip(self) -> Any:
        return self

    def vertical_flip(self) -> Any:
        return self

    # TODO: We have to ignore override mypy error as there is torch.Tensor built-in deprecated op: Tensor.resize
    # https://github.com/pytorch/pytorch/blob/e8727994eb7cdb2ab642749d6549bc497563aa06/torch/_tensor.py#L588-L593
    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> Any:
        return self

    def crop(self, top: int, left: int, height: int, width: int) -> Any:
        return self

    def center_crop(self, output_size: List[int]) -> Any:
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
    ) -> Any:
        return self

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        padding_mode: str = "constant",
    ) -> Any:
        return self

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> Any:
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
    ) -> Any:
        return self

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> Any:
        return self

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> Any:
        return self

    def adjust_brightness(self, brightness_factor: float) -> Any:
        return self

    def adjust_saturation(self, saturation_factor: float) -> Any:
        return self

    def adjust_contrast(self, contrast_factor: float) -> Any:
        return self

    def adjust_sharpness(self, sharpness_factor: float) -> Any:
        return self

    def adjust_hue(self, hue_factor: float) -> Any:
        return self

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Any:
        return self

    def posterize(self, bits: int) -> Any:
        return self

    def solarize(self, threshold: float) -> Any:
        return self

    def autocontrast(self) -> Any:
        return self

    def equalize(self) -> Any:
        return self

    def invert(self) -> Any:
        return self

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Any:
        return self
