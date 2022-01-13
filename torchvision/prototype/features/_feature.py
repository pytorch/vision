from typing import Any, Callable, cast, Dict, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar

import torch
from torch._C import _TensorBase, DisableTorchFunction


F = TypeVar("F", bound="Feature")


class MetaData:
    EMPTY = object()

    def __init__(self, value, fallback=EMPTY):
        self.value = value
        self.fallback = fallback


class Feature(torch.Tensor):
    _META_ATTRS: Set[str] = set()
    _metadata: Dict[str, Any]
    _KERNELS: Dict[Callable, Callable]

    def __init_subclass__(cls):
        # In order to help static type checkers, we require subclasses of `Feature` to add the metadata attributes
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
            setattr(cls, name, property(lambda self, name=name: self._metadata[name]))

        cls._KERNELS = {}

    def __new__(cls, data, *, dtype=None, device=None):
        feature = torch.Tensor._make_subclass(
            cast(_TensorBase, cls),
            cls._to_tensor(data, dtype=dtype, device=device),
            # requires_grad
            False,
        )
        feature._metadata = dict()
        return feature

    @classmethod
    def _to_tensor(self, data: Any, *, dtype, device) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype, device=device)

    @classmethod
    def new_like(cls, other, data, *, dtype=None, device=None, **metadata):
        for name in cls._META_ATTRS:
            metadata.setdefault(name, getattr(other, name))
        return cls(data, dtype=dtype or other.dtype, device=device or other.device, **metadata)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        kwargs = kwargs or dict()
        if cls is not Feature and func in cls._KERNELS:
            return cls._KERNELS[func](*args, **kwargs)

        with DisableTorchFunction():
            output = func(*args, **kwargs)

        if func is torch.Tensor.clone:
            return cls.new_like(args[0], output)

        return output

    def __repr__(self):
        return torch.Tensor.__repr__(self).replace("tensor", type(self).__name__)
