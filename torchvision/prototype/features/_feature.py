from typing import Any, Callable, cast, Dict, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar

import torch
from torch._C import _TensorBase, DisableTorchFunction
from torchvision.prototype.utils._internal import add_suggestion


F = TypeVar("F", bound="Feature")


_EMPTY = object()


class Feature(torch.Tensor):
    _META_ATTRS: Set[str] = set()
    _meta_data: Dict[str, Any]
    _KERNELS: Dict[Callable, Callable] = {}

    def __init_subclass__(cls):
        # In order to help static type checkers, we require subclasses of `Feature` to add the meta data attributes
        # as static class annotations:
        #
        # >>> class Foo(Feature):
        # ...     bar: str
        # ...     baz: Optional[str]
        #
        # Internally, this information is used twofold:
        #
        # 1. A class annotation is contained in `cls.__annotations__` but not in `cls.__dict__`. We use this difference
        #    to automatically detect the meta data attributes and expose them as `@property`'s for convenient runtime
        #    access. This happens in this method.
        # 2. The information extracted in 1. is also used at creation (`__new__`) to perform an input parsing for
        #    unknown arguments.
        meta_attrs = {attr for attr in cls.__annotations__.keys() - cls.__dict__.keys() if not attr.startswith("_")}
        for super_cls in cls.__mro__[1:]:
            if super_cls is Feature:
                break

            meta_attrs.update(super_cls._META_ATTRS)

        cls._META_ATTRS = meta_attrs
        for name in meta_attrs:
            setattr(cls, name, property(lambda self, name=name: self._meta_data[name]))

    def __new__(cls, data, *, dtype=None, device=None, like=None, **kwargs):
        unknown_meta_attrs = kwargs.keys() - cls._META_ATTRS
        if unknown_meta_attrs:
            unknown_meta_attr = sorted(unknown_meta_attrs)[0]
            raise TypeError(
                add_suggestion(
                    f"{cls.__name__}() got unexpected keyword '{unknown_meta_attr}'.",
                    word=unknown_meta_attr,
                    possibilities=sorted(cls._META_ATTRS),
                )
            )

        if like is not None:
            dtype = dtype or like.dtype
            device = device or like.device
        data = cls._to_tensor(data, dtype=dtype, device=device)
        requires_grad = False
        self = torch.Tensor._make_subclass(cast(_TensorBase, cls), data, requires_grad)

        meta_data = {}
        for name in cls._META_ATTRS:
            if name in kwargs:
                value = kwargs[name]
            elif like is not None:
                value = getattr(like, name)
            else:
                value = _EMPTY
            meta_data[name] = value
        meta_data = cls._prepare_meta_data(data, meta_data)
        for key, value in meta_data.items():
            if value is _EMPTY:
                raise TypeError(
                    f"{cls.__name__}() is missing a required argument: '{key}'. Either pass is explicitly, "
                    f"or use the 'like' parameter to extract it from an object of the same type."
                )
        self._meta_data = meta_data

        return self

    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        return torch.as_tensor(data, dtype=dtype, device=device)

    @classmethod
    def _prepare_meta_data(cls, data: torch.Tensor, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        return meta_data

    _TORCH_FUNCTION_ALLOW_LIST = {
        torch.Tensor.clone,
    }

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        kwargs = kwargs or dict()
        if func in cls._KERNELS:
            return cls._KERNELS[func](*args, **kwargs)

        with DisableTorchFunction():
            output = func(*args, **(kwargs or dict()))

        if func not in cls._TORCH_FUNCTION_ALLOW_LIST:
            return output

        return cls(output, like=args[0])

    def __repr__(self):
        return torch.Tensor.__repr__(self).replace("tensor", type(self).__name__)
