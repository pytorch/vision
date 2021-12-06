from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def grid_sample(img: Tensor, absolute_grid: Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)


def make_coords_grid(batch_size: int, h: int, w: int):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)


def upsample_flow(flow, up_mask: Optional[Tensor] = None):
    """Upsample flow by a factor of 8.

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights. See paper page 8 and appendix B.
    Note that in appendix B the picture assumes a downsample factor of 4 instead of 8.
    """
    batch_size, _, h, w = flow.shape
    new_h, new_w = h * 8, w * 8

    if up_mask is None:
        return 8 * F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)

    up_mask = up_mask.view(batch_size, 1, 9, 8, 8, h, w)
    up_mask = torch.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

    upsampled_flow = F.unfold(8 * flow, kernel_size=3, padding=1).view(batch_size, 2, 9, 1, 1, h, w)
    upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)

    return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, 2, new_h, new_w)
