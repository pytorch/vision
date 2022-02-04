import torch
from torchvision.prototype.features import BoundingBoxFormat, ColorSpace
from torchvision.transforms.functional_tensor import rgb_to_grayscale as _rgb_to_grayscale


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xyxy = xywh.clone()  # Nice. Do we have tests to ensure that no transform does an in-place modification of input?
    xyxy[..., 2:] += xyxy[..., :2]
    return xyxy


def _xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    xywh = xyxy.clone()
    xywh[..., 2:] -= xywh[..., :2]
    return xywh


def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.unbind(xyxy, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def convert_bounding_box_format(
    bounding_box: torch.Tensor, *, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat
) -> torch.Tensor:
    if new_format == old_format:
        return bounding_box

    if old_format == BoundingBoxFormat.XYWH:
        bounding_box = _xywh_to_xyxy(bounding_box)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _cxcywh_to_xyxy(bounding_box)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_box = _xyxy_to_xywh(bounding_box)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _xyxy_to_cxcywh(bounding_box)

    return bounding_box


def _grayscale_to_rgb(grayscale: torch.Tensor) -> torch.Tensor:
    return grayscale.expand(3, 1, 1)


def convert_color_space(image: torch.Tensor, old_color_space: ColorSpace, new_color_space: ColorSpace) -> torch.Tensor:
    if new_color_space == old_color_space:
        return image

    if old_color_space == ColorSpace.GRAYSCALE:
        image = _grayscale_to_rgb(image)

    if new_color_space == ColorSpace.GRAYSCALE:
        image = _rgb_to_grayscale(image)

    return image
