from typing import Any, cast, Dict, Set, TypeVar, Union, Optional, Type, Callable, Tuple, Sequence, Mapping

import torch
from torch._C import _TensorBase, DisableTorchFunction


F = TypeVar("F", bound="Feature")


class Feature(torch.Tensor):
    _META_ATTRS: Set[str] = set()
    _metadata: Dict[str, Any]

    def __init_subclass__(cls) -> None:
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

            meta_attrs.update(cast(Type[Feature], super_cls)._META_ATTRS)

        cls._META_ATTRS = meta_attrs
        for name in meta_attrs:
            setattr(cls, name, property(cast(Callable[[F], Any], lambda self, name=name: self._metadata[name])))

    def __new__(
        cls: Type[F],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> F:
        if isinstance(device, str):
            device = torch.device(device)
        feature = cast(
            F,
            torch.Tensor._make_subclass(
                cast(_TensorBase, cls),
                cls._to_tensor(data, dtype=dtype, device=device),
                # requires_grad
                False,
            ),
        )
        feature._metadata = dict()
        return feature

    @classmethod
    def _to_tensor(self, data: Any, *, dtype: Optional[torch.dtype], device: Optional[torch.device]) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype, device=device)

    @classmethod
    def new_like(
        cls: Type[F],
        other: F,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
        **metadata: Any,
    ) -> F:
        _metadata = other._metadata.copy()
        _metadata.update(metadata)
        return cls(data, dtype=dtype or other.dtype, device=device or other.device, **_metadata)

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

    def __repr__(self) -> str:
        return cast(str, torch.Tensor.__repr__(self)).replace("tensor", type(self).__name__)
