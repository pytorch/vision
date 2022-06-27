from typing import Any, cast, TypeVar, Union, Optional, Type, Callable, Tuple, Sequence, Mapping

import torch
from torch._C import _TensorBase, DisableTorchFunction


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
        feature = cast(
            F,
            torch.Tensor._make_subclass(
                cast(_TensorBase, cls),
                torch.as_tensor(data, dtype=dtype, device=device),  # type: ignore[arg-type]
                requires_grad,
            ),
        )

        # To avoid circular dependency between features and transforms
        from ..transforms import functional

        feature._F = functional

        return feature

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

    def horizontal_flip(self):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def vertical_flip(self):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def resize(self, size, *, interpolation, max_size, antialias):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def center_crop(self, output_size):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def resized_crop(self, top, left, height, width, *, size, interpolation, antialias):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def pad(self, padding, *, fill, padding_mode):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def rotate(self, angle, *, interpolation, expand, fill, center):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def affine(self, angle, *, translate, scale, shear, interpolation, fill, center):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_brightness(self, brightness_factor: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_saturation(self, saturation_factor: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_contrast(self, contrast_factor: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_sharpness(self, sharpness_factor: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_hue(self, hue_factor: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def adjust_gamma(self, gamma: float, gain: float = 1):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def posterize(self, bits: int):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def solarize(self, threshold: float):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def autocontrast(self):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def equalize(self):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def equalize(self):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def erase(self, i, j, h, w, v):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def mixup(self, lam):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self

    def cutmix(self, *, box, lam_adjusted):
        # Just output itself
        # How dangerous to do this instead of raising an error ?
        return self
