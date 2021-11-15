from typing import Tuple, cast, TypeVar, Set, Dict, Any, Callable, Optional, Mapping, Type, Sequence

import torch
from torch._C import _TensorBase, DisableTorchFunction
from torchvision.prototype.utils._internal import add_suggestion


F = TypeVar("F", bound="Feature")


DEFAULT = object()


class Feature(torch.Tensor):
    _META_ATTRS: Set[str]
    _meta_data: Dict[str, Any]

    def __init_subclass__(cls):
        if not hasattr(cls, "_META_ATTRS"):
            cls._META_ATTRS = {
                attr for attr in cls.__annotations__.keys() - cls.__dict__.keys() if not attr.startswith("_")
            }

        for attr in cls._META_ATTRS:
            if not hasattr(cls, attr):
                setattr(cls, attr, property(lambda self, attr=attr: self._meta_data[attr]))

    def __new__(cls, data, *, dtype=None, device=None, like=None, **kwargs):
        unknown_meta_attrs = kwargs.keys() - cls._META_ATTRS
        if unknown_meta_attrs:
            unknown_meta_attr = sorted(unknown_meta_attrs)[0]
            raise TypeError(
                add_suggestion(
                    f"{cls.__name__}() got unexpected keyword '{unknown_meta_attr}'.",
                    word=unknown_meta_attr,
                    possibilities=cls._META_ATTRS,
                )
            )

        if like is not None:
            dtype = dtype or like.dtype
            device = device or like.device
        data = cls._to_tensor(data, dtype=dtype, device=device)
        requires_grad = False
        self = torch.Tensor._make_subclass(cast(_TensorBase, cls), data, requires_grad)

        meta_data = dict()
        for attr, (explicit, fallback) in cls._parse_meta_data(**kwargs).items():
            if explicit is not DEFAULT:
                value = explicit
            elif like is not None:
                value = getattr(like, attr)
            else:
                value = fallback(data) if callable(fallback) else fallback
            meta_data[attr] = value
        self._meta_data = meta_data

        return self

    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        return torch.as_tensor(data, dtype=dtype, device=device)

    @classmethod
    def _parse_meta_data(cls) -> Dict[str, Tuple[Any, Any]]:
        return dict()

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

    def __repr__(self):
        return torch.Tensor.__repr__(self).replace("tensor", type(self).__name__)
