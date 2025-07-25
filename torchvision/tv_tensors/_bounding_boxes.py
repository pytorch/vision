from __future__ import annotations

from collections.abc import Mapping, Sequence

from enum import Enum
from typing import Any, Optional

import torch
from torch.utils._pytree import tree_flatten

from ._tv_tensor import TVTensor


class BoundingBoxFormat(Enum):
    """Coordinate format of a bounding box.

    Available formats are:

    * ``XYXY``: bounding box represented via corners; x1, y1 being top left;
      x2, y2 being bottom right.
    * ``XYWH``: bounding box represented via corner, width and height; x1, y1
      being top left; w, h being width and height.
    * ``CXCYWH``: bounding box represented via centre, width and height; cx,
      cy being center of box; w, h being width and height.
    * ``XYWHR``: rotated boxes represented via corner, width and height; x1, y1
      being top left; w, h being width and height. r is rotation angle in
      degrees.
    * ``CXCYWHR``: rotated boxes represented via center, width and height; cx,
      cy being center of box; w, h being width and height. r is rotation angle
      in degrees.
    * ``XYXYXYXY``: rotated boxes represented via corners; x1, y1 being top
      left; x2, y2 being top right; x3, y3 being bottom right; x4, y4 being
      bottom left.
    """

    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"
    XYWHR = "XYWHR"
    CXCYWHR = "CXCYWHR"
    XYXYXYXY = "XYXYXYXY"


# TODO: Once torchscript supports Enums with staticmethod
# this can be put into BoundingBoxFormat as staticmethod
def is_rotated_bounding_format(format: BoundingBoxFormat | str) -> bool:
    if isinstance(format, BoundingBoxFormat):
        return (
            format == BoundingBoxFormat.XYWHR
            or format == BoundingBoxFormat.CXCYWHR
            or format == BoundingBoxFormat.XYXYXYXY
        )
    elif isinstance(format, str):
        return format in ("XYWHR", "CXCYWHR", "XYXYXYXY")
    else:
        raise ValueError(f"format should be str or BoundingBoxFormat, got {type(format)}")


# This should ideally be a Literal, but torchscript fails.
CLAMPING_MODE_TYPE = Optional[str]


class BoundingBoxes(TVTensor):
    """:class:`torch.Tensor` subclass for bounding boxes with shape ``[N, K]``.

    .. note::
        Support for rotated bounding boxes was released in TorchVision 0.23 and
        is currently a BETA feature. We don't expect the API to change, but
        there may be some rare edge-cases. If you find any issues, please report
        them on our bug tracker:
        https://github.com/pytorch/vision/issues?q=is:open+is:issue

    Where ``N`` is the number of bounding boxes
    and ``K`` is 4 for unrotated boxes, and 5 or 8 for rotated boxes.

    .. note::
        There should be only one :class:`~torchvision.tv_tensors.BoundingBoxes`
        instance per sample e.g. ``{"img": img, "bbox": BoundingBoxes(...)}``,
        although one :class:`~torchvision.tv_tensors.BoundingBoxes` object can
        contain multiple bounding boxes.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BoundingBoxFormat, str): Format of the bounding box.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        clamping_mode: The clamping mode to use when applying transforms that may result in bounding boxes
            partially outside of the image. Possible values are: "soft", "hard", or ``None``. Read more in :ref:`clamping_mode_tuto`.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BoundingBoxFormat
    canvas_size: tuple[int, int]
    clamping_mode: CLAMPING_MODE_TYPE

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, format: BoundingBoxFormat | str, canvas_size: tuple[int, int], clamping_mode: CLAMPING_MODE_TYPE = "soft", check_dims: bool = True) -> BoundingBoxes:  # type: ignore[override]
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")
        if clamping_mode is not None and clamping_mode not in ("hard", "soft"):
            raise ValueError(f"clamping_mode must be None, hard or soft, got {clamping_mode}.")

        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]

        bounding_boxes = tensor.as_subclass(cls)
        bounding_boxes.format = format
        bounding_boxes.canvas_size = canvas_size
        bounding_boxes.clamping_mode = clamping_mode
        return bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: BoundingBoxFormat | str,
        canvas_size: tuple[int, int],
        clamping_mode: CLAMPING_MODE_TYPE = "soft",
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> BoundingBoxes:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if not torch.is_floating_point(tensor) and is_rotated_bounding_format(format):
            raise ValueError(f"Rotated bounding boxes should be floating point tensors, got {tensor.dtype}.")
        return cls._wrap(tensor, format=format, canvas_size=canvas_size, clamping_mode=clamping_mode)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> BoundingBoxes:
        # If there are BoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, BoundingBoxes))
        format, canvas_size, clamping_mode = (
            first_bbox_from_args.format,
            first_bbox_from_args.canvas_size,
            first_bbox_from_args.clamping_mode,
        )

        if isinstance(output, torch.Tensor) and not isinstance(output, BoundingBoxes):
            output = BoundingBoxes._wrap(
                output, format=format, canvas_size=canvas_size, clamping_mode=clamping_mode, check_dims=False
            )
        elif isinstance(output, (tuple, list)):
            # This branch exists for chunk() and unbind()
            output = type(output)(
                BoundingBoxes._wrap(
                    part, format=format, canvas_size=canvas_size, clamping_mode=clamping_mode, check_dims=False
                )
                for part in output
            )
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, canvas_size=self.canvas_size, clamping_mode=self.clamping_mode)
