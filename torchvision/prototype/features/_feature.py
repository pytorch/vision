from typing import Tuple, cast, TypeVar, Set, Dict, Any, Callable, Optional, Mapping, Type, Sequence

import torch
from torch._C import _TensorBase, DisableTorchFunction
from torchvision.prototype.utils._internal import add_suggestion


F = TypeVar("F", bound="Feature")


DEFAULT = object()


class Feature(torch.Tensor):
    _META_ATTRS: Set[str] = set()
    _meta_data: Dict[str, Any]

    def __init_subclass__(cls):
        # In order to help static type checkers, we require subclasses of `Feature` add the meta data attributes
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
        for attr in meta_attrs:
            setattr(cls, attr, property(lambda self, attr=attr: self._meta_data[attr]))

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
