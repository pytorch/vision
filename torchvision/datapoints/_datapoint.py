from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

import PIL.Image
import torch


D = TypeVar("D", bound="Datapoint")
_FillType = Union[int, float, Sequence[int], Sequence[float], None]
_FillTypeJIT = Optional[List[float]]


class Datapoint(torch.Tensor):
    """[Beta] Base class for all datapoints.

    You probably don't want to use this class unless you're defining your own
    custom Datapoints. See
    :ref:`sphx_glr_auto_examples_plot_custom_datapoints.py` for details.
    """

    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    # We have to override a few method to make sure the meta-data is preserved on them.
    @classmethod
    def wrap_like(cls: Type[D], other: D, tensor: torch.Tensor) -> D:
        return tensor.as_subclass(cls)

    def clone(self):
        return type(self).wrap_like(self, super().clone())

    def to(self, *args, **kwargs):
        return type(self).wrap_like(self, super().to(*args, **kwargs))

    def detach(self):
        return type(self).wrap_like(self, super().detach())

    def requires_grad_(self, requires_grad: bool = True):
        return type(self).wrap_like(self, super().requires_grad_(requires_grad))

    def _make_repr(self, **kwargs: Any) -> str:
        # This is a poor man's implementation of the proposal in https://github.com/pytorch/pytorch/issues/76532.
        # If that ever gets implemented, remove this in favor of the solution on the `torch.Tensor` class.
        extra_repr = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{super().__repr__()[:-1]}, {extra_repr})"

    def __deepcopy__(self: D, memo: Dict[int, Any]) -> D:
        # We need to detach first, since a plain `Tensor.clone` will be part of the computation graph, which does
        # *not* happen for `deepcopy(Tensor)`. A side-effect from detaching is that the `Tensor.requires_grad`
        # attribute is cleared, so we need to refill it before we return.
        # Note: We don't explicitly handle deep-copying of the metadata here. The only metadata we currently have is
        # `BoundingBoxes.format` and `BoundingBoxes.canvas_size`, which are immutable and thus implicitly deep-copied by
        # `BoundingBoxes.clone()`.
        return self.detach().clone().requires_grad_(self.requires_grad)  # type: ignore[return-value]


_InputType = Union[torch.Tensor, PIL.Image.Image, Datapoint]
_InputTypeJIT = torch.Tensor
