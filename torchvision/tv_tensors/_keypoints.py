from __future__ import annotations

from typing import Any, Mapping, MutableSequence, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch
from torch.utils._pytree import tree_flatten

from ._tv_tensor import TVTensor


class KeyPoints(TVTensor):
    """:class:`torch.Tensor` subclass for tensors with shape ``[..., 2]`` that represent points in an image.

    Each point is represented by its XY coordinates.

    KeyPoints can be converted from :class:`torchvision.tv_tensors.BoundingBoxes`
    by :func:`torchvision.transforms.v2.functional.convert_box_to_points`.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    canvas_size: Tuple[int, int]

    def __new__(
        cls,
        data: Any,
        *,
        canvas_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ):
        tensor: torch.Tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] != 2:
            raise ValueError(f"Expected a tensor of shape (..., 2), not {tensor.shape}")
        points = tensor.as_subclass(cls)
        points.canvas_size = canvas_size
        return points

    if TYPE_CHECKING:
        # EVIL: Just so that MYPY+PYLANCE+others stop shouting that everything is wrong when initializeing the TVTensor
        # Not read or defined at Runtime (only at linting time).
        # TODO: BOUNDING BOXES needs something similar
        def __init__(
            self,
            data: Any,
            *,
            canvas_size: Tuple[int, int],
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[torch.device, str, int]] = None,
            requires_grad: Optional[bool] = None,
        ):
            ...

    @classmethod
    def _wrap_output(
        cls,
        output: Any,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        # Mostly copied over from the BoundingBoxes TVTensor, minor improvements.
        # This copies over the metadata.
        # For BoundingBoxes, that included format, but we only support one format here !
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, KeyPoints))
        canvas_size: Tuple[int, int] = first_bbox_from_args.canvas_size

        if isinstance(output, torch.Tensor) and not isinstance(output, KeyPoints):
            output = KeyPoints(output, canvas_size=canvas_size)
        elif isinstance(output, tuple):
            # NB: output is checked against sequence because it has already been checked against Tensor
            # Since a Tensor is a sequence of Tensor, had it not been the case, we may have had silent
            # or complex errors
            output = tuple(KeyPoints(part, canvas_size=canvas_size) for part in output)
        elif isinstance(output, MutableSequence):
            for i, part in enumerate(output):
                output[i] = KeyPoints(part, canvas_size=canvas_size)
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(canvas_size=self.canvas_size)
