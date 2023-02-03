from __future__ import annotations

from typing import Any, Optional, Union

import torch

from ._datapoint import Datapoint


class Label(Datapoint):
    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Label:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Label:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: Label,
        tensor: torch.Tensor,
    ) -> Label:
        return cls._wrap(tensor)


class OneHotLabel(Datapoint):
    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> OneHotLabel:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> OneHotLabel:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: OneHotLabel,
        tensor: torch.Tensor,
    ) -> OneHotLabel:
        return cls._wrap(tensor)
