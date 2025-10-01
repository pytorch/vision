from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch.utils._pytree import tree_flatten

from ._tv_tensor import TVTensor
from ._bounding_boxes import CLAMPING_MODE_TYPE


class KeyPoints(TVTensor):
    """:class:`torch.Tensor` subclass for tensors with shape ``[..., 2]`` that represent points in an image.

    .. note::
        Support for keypoints was released in TorchVision 0.23 and is currently
        a BETA feature. We don't expect the API to change, but there may be some
        rare edge-cases. If you find any issues, please report them on our bug
        tracker: https://github.com/pytorch/vision/issues?q=is:open+is:issue
        Each point is represented by its X and Y coordinates along the width and
        height dimensions, respectively.

    Each point is represented by its X and Y coordinates along the width and height dimensions, respectively.

    KeyPoints may represent any object that can be represented by sequences of 2D points:

    - `Polygonal chains <https://en.wikipedia.org/wiki/Polygonal_chain>`_,
      including polylines, Bézier curves, etc., which can be of shape
      ``[N_chains, N_points, 2]``.
    - Polygons, which can be of shape ``[N_polygons, N_points, 2]``.
    - Skeletons, which can be of shape ``[N_skeletons, N_bones, 2, 2]`` for
      pose-estimation models.

    .. note::
        Like for :class:`torchvision.tv_tensors.BoundingBoxes`, there should
        only be a single instance of the
        :class:`torchvision.tv_tensors.KeyPoints` class per sample e.g.
        ``{"img": img, "poins_of_interest": KeyPoints(...)}``, although one
        :class:`torchvision.tv_tensors.KeyPoints` object can contain multiple
        key points

    Args:
        data: Any data that can be turned into a tensor with
            :func:`torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding
            image or video.
        clamping_mode: The clamping mode to use when applying transforms that may result in key points
            outside of the image. Possible values are: "soft", "hard", or ``None``. Read more in :ref:`clamping_mode_tuto`.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If
            omitted, will be inferred from ``data``.
        device (torch.device, optional): Desired device of the bounding box. If
            omitted and ``data`` is a :class:`torch.Tensor`, the device is taken
            from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record
            operations on the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the value is taken from it. Otherwise,
            defaults to ``False``.
    """

    canvas_size: tuple[int, int]
    clamping_mode: CLAMPING_MODE_TYPE

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, canvas_size: tuple[int, int], clamping_mode: CLAMPING_MODE_TYPE = "soft", check_dims: bool = True) -> KeyPoints:  # type: ignore[override]
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] != 2:
                raise ValueError(f"Expected a tensor of shape (..., 2), not {tensor.shape}")
        if clamping_mode is not None and clamping_mode not in ("hard", "soft"):
            raise ValueError(f"clamping_mode must be None, hard or soft, got {clamping_mode}.")
        points = tensor.as_subclass(cls)
        points.canvas_size = canvas_size
        points.clamping_mode = clamping_mode
        return points

    def __new__(
        cls,
        data: Any,
        *,
        canvas_size: tuple[int, int],
        clamping_mode: CLAMPING_MODE_TYPE = "soft",
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> KeyPoints:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size, clamping_mode=clamping_mode)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> KeyPoints:
        # Similar to BoundingBoxes._wrap_output(), see comment there.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_keypoints_from_args = next(x for x in flat_params if isinstance(x, KeyPoints))
        canvas_size, clamping_mode = first_keypoints_from_args.canvas_size, first_keypoints_from_args.clamping_mode

        if isinstance(output, torch.Tensor) and not isinstance(output, KeyPoints):
            output = KeyPoints._wrap(output, canvas_size=canvas_size, clamping_mode=clamping_mode, check_dims=False)
        elif isinstance(output, (tuple, list)):
            # This branch exists for chunk() and unbind()
            output = type(output)(KeyPoints._wrap(part, canvas_size=canvas_size, clamping_mode=clamping_mode, check_dims=False) for part in output)
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(canvas_size=self.canvas_size, clamping_mode=self.clamping_mode)
