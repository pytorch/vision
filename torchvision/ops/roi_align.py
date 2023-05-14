from typing import List, Union

import torch
import torch._dynamo
import torch.fx
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _has_ops

from ..utils import _log_api_usage_once
from ._utils import check_roi_boxes_shape, convert_boxes_to_roi_format


# NB: all tensor inputs
def _bilinear_interpolate(input, roi_batch_ind, c, height, width, y, x, ymask, xmask):
    from functorch.dim import dims

    # deal with inverse element out of feature map boundary
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx

    # do bilinear interpolation, but respect the masking!
    # TODO: It's possible the masking here is unnecessary if y and
    # x were clamped appropriately; hard to tell
    def masked_index(y, x):
        if ymask is not None:
            assert xmask is not None
            y = torch.where(ymask, y, 0)
            x = torch.where(xmask, x, 0)
        return input[roi_batch_ind, c, y, x]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx;

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

# TODO: this doesn't actually cache
# TODO: main library should make this easier to do
def maybe_cast(tensor):
    if torch.is_autocast_enabled() and tensor.is_cuda and tensor.dtype != torch.double:
        return tensor.float()
    else:
        return tensor

# This is a slow but pure Python and differentiable implementation of
# roi_align.  It potentially is a good basis for Inductor compilation
# (but I have not benchmarked it) but today it is solely used for the
# fact that its backwards can be implemented deterministically.
#
# It is transcribed directly off of the roi_align CUDA kernel, see
# https://dev-discuss.pytorch.org/t/a-pure-python-implementation-of-roi-align-that-looks-just-like-its-cuda-kernel/1266
@torch._dynamo.allow_in_graph
def _roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    from functorch.dim import dims

    orig_dtype = input.dtype

    input = maybe_cast(input)
    rois = maybe_cast(rois)

    _, _, height, width = input.size()

    n, c, ph, pw = dims(4)
    ph.size = pooled_height
    pw.size = pooled_width
    offset_rois = rois[n]
    roi_batch_ind = offset_rois[0].int()
    offset = 0.5 if aligned else 0.0
    roi_start_w = offset_rois[1] * spatial_scale - offset
    roi_start_h = offset_rois[2] * spatial_scale - offset
    roi_end_w = offset_rois[3] * spatial_scale - offset
    roi_end_h = offset_rois[4] * spatial_scale - offset

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)
        roi_height = torch.clamp(roi_height, min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    exact_sampling = sampling_ratio > 0

    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / pooled_width)

    iy, ix = dims(2)

    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy.size = roi_bin_grid_h
        ix.size = roi_bin_grid_w
        ymask = None
        xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        # When doing adaptive sampling, the number of samples we need to do
        # is data-dependent based on how big the ROIs are.  This is a bit
        # awkward because first-class dims can't actually handle this.
        # So instead, we inefficiently suppose that we needed to sample ALL
        # the points and mask out things that turned out to be unnecessary
        iy.size = height
        ix.size = width
        ymask = iy < roi_bin_grid_h
        xmask = ix < roi_bin_grid_w

    y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
    val = _bilinear_interpolate(input, roi_batch_ind, c, height, width, y, x, ymask, xmask)

    # Mask out samples that weren't actually adaptively needed
    if not exact_sampling:
        val = torch.where(ymask, val, 0)
        val = torch.where(xmask, val, 0)

    output = val.sum((iy, ix))
    output /= count

    output = output.to(orig_dtype)

    return output.order(n, c, ph, pw)


@torch.fx.wrap
def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the box coordinates to
            the input coordinates. For example, if your boxes are defined on the scale
            of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            the original image), you'll want to set this to 0.5. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(roi_align)
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    if not torch.jit.is_scripting():
        if not _has_ops() or (torch.are_deterministic_algorithms_enabled() and input.is_cuda):
            return _roi_align(
                input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
            )
    return torch.ops.torchvision.roi_align(
        input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
    )


class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """

    def __init__(
        self,
        output_size: BroadcastingList2[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}"
            f", spatial_scale={self.spatial_scale}"
            f", sampling_ratio={self.sampling_ratio}"
            f", aligned={self.aligned}"
            f")"
        )
        return s
