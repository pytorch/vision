import enum
from typing import Any, Sequence, Mapping, Optional, Tuple, Type, Callable, TypeVar, Union, cast

import torch
from torch._C import DisableTorchFunction, _TensorBase
from torchvision.prototype.datasets.utils._internal import FrozenMapping

__all__ = ["Feature", "ColorSpace", "Image", "Label"]

T = TypeVar("T", bound="Feature")


class Feature(torch.Tensor):
    _meta_data: FrozenMapping[str, Any]

    def __new__(
        cls: Type[T],
        data: Any,
        *,
        like: Optional[T] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        meta_data: FrozenMapping[str, Any] = FrozenMapping(),
    ) -> T:
        if like is not None:
            dtype = dtype or like.dtype
            device = device or like.device
        data = torch.as_tensor(data, dtype=dtype, device=device)
        requires_grad = False
        self = cast(T, torch.Tensor._make_subclass(cast(_TensorBase, cls), data, requires_grad))

        _meta_data = dict(like._meta_data) if like is not None else dict()
        _meta_data.update(meta_data)
        self._meta_data = FrozenMapping(_meta_data)

        for name in self._meta_data:
            setattr(cls, name, property(lambda self: self._meta_data[name]))

        return self

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        with DisableTorchFunction():
            output = func(*args, **(kwargs or dict()))
        if func is not torch.Tensor.clone:
            return output

        return cls(output, like=args[0])

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", type(self).__name__)


class ColorSpace(enum.Enum):
    # this is just for test purposes
    _SENTINEL = -1
    OTHER = 0
    GRAYSCALE = 1
    RGB = 3


class Image(Feature):
    color_space: ColorSpace

    def __new__(
        cls,
        data: Any,
        *,
        like: Optional["Image"] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        color_space: Optional[Union[str, ColorSpace]] = None,
    ) -> "Image":
        if color_space is None:
            color_space = cls.guess_color_space(data) if data is not None else ColorSpace.OTHER
        elif isinstance(color_space, str):
            color_space = ColorSpace[color_space.upper()]

        meta_data: FrozenMapping[str, Any] = FrozenMapping(color_space=color_space)

        return Feature.__new__(cls, data, like=like, dtype=dtype, device=device, meta_data=meta_data)

    @staticmethod
    def guess_color_space(image: torch.Tensor) -> ColorSpace:
        if image.ndim < 2:
            return ColorSpace.OTHER
        elif image.ndim == 2:
            return ColorSpace.GRAYSCALE

        num_channels = image.shape[-3]
        if num_channels == 1:
            return ColorSpace.GRAYSCALE
        elif num_channels == 3:
            return ColorSpace.RGB
        else:
            return ColorSpace.OTHER


class Label(Feature):
    category: Optional[str]

    def __new__(
        cls,
        data: Any,
        *,
        like: Optional["Label"] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        category: Optional[str] = None,
    ) -> "Label":
        return Feature.__new__(
            cls, data, like=like, dtype=dtype, device=device, meta_data=FrozenMapping(category=category)
        )
