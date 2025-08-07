import torch
from torch import Tensor


def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes


def _box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes


def _box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    x, y, w, h = boxes.unbind(-1)
    boxes = torch.stack([x, y, x + w, y + h], dim=-1)
    return boxes


def _box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    boxes = torch.stack((x1, y1, w, h), dim=-1)
    return boxes


def _box_cxcywhr_to_xywhr(boxes: Tensor) -> Tensor:
    """
    Converts rotated bounding boxes from (cx, cy, w, h, r) format to (x1, y1, w, h, r) format.
    (cx, cy) refers to center of bounding box
    (w, h) refers to width and height of rotated bounding box
    (x1, y1) refers to top left of rotated bounding box
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan
    Args:
        boxes (Tensor[N, 5]): boxes in (cx, cy, w, h, r) format which will be converted.

    Returns:
        boxes (Tensor(N, 5)): rotated boxes in (x1, y1, w, h, r) format.
    """
    dtype = boxes.dtype
    need_cast = not boxes.is_floating_point()
    cx, cy, w, h, r = boxes.unbind(-1)
    r_rad = r * torch.pi / 180.0
    cos, sin = torch.cos(r_rad), torch.sin(r_rad)

    x1 = cx - w / 2 * cos - h / 2 * sin
    y1 = cy - h / 2 * cos + w / 2 * sin
    boxes = torch.stack((x1, y1, w, h, r), dim=-1)

    if need_cast:
        boxes.round_()
        boxes = boxes.to(dtype)
    return boxes


def _box_xywhr_to_cxcywhr(boxes: Tensor) -> Tensor:
    """
    Converts rotated bounding boxes from (x1, y1, w, h, r) format to (cx, cy, w, h, r) format.
    (x1, y1) refers to top left of rotated bounding box
    (w, h) refers to width and height of rotated bounding box
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan
    Args:
        boxes (Tensor[N, 5]): rotated boxes in (x1, y1, w, h, r) format which will be converted.

    Returns:
        boxes (Tensor[N, 5]): rotated boxes in (cx, cy, w, h, r) format.
    """
    dtype = boxes.dtype
    need_cast = not boxes.is_floating_point()
    x1, y1, w, h, r = boxes.unbind(-1)
    r_rad = r * torch.pi / 180.0
    cos, sin = torch.cos(r_rad), torch.sin(r_rad)

    cx = x1 + w / 2 * cos + h / 2 * sin
    cy = y1 - w / 2 * sin + h / 2 * cos

    boxes = torch.stack([cx, cy, w, h, r], dim=-1)
    if need_cast:
        boxes.round_()
        boxes = boxes.to(dtype)
    return boxes


def _box_xywhr_to_xyxyxyxy(boxes: Tensor) -> Tensor:
    """
    Converts rotated bounding boxes from (x1, y1, w, h, r) format to (x1, y1, x2, y2, x3, y3, x4, y4) format.
    (x1, y1) refer to top left of bounding box
    (w, h) are width and height of the rotated bounding box
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    (x1, y1) refer to top left of rotated bounding box
    (x2, y2) refer to top right of rotated bounding box
    (x3, y3) refer to bottom right of rotated bounding box
    (x4, y4) refer to bottom left ofrotated bounding box
    Args:
        boxes (Tensor[N, 5]): rotated boxes in (cx, cy, w, h, r) format which will be converted.

    Returns:
        boxes (Tensor(N, 8)): rotated boxes in (x1, y1, x2, y2, x3, y3, x4, y4) format.
    """
    dtype = boxes.dtype
    need_cast = not boxes.is_floating_point()
    x1, y1, w, h, r = boxes.unbind(-1)
    r_rad = r * torch.pi / 180.0
    cos, sin = torch.cos(r_rad), torch.sin(r_rad)

    x2 = x1 + w * cos
    y2 = y1 - w * sin
    x3 = x2 + h * sin
    y3 = y2 + h * cos
    x4 = x1 + h * sin
    y4 = y1 + h * cos

    boxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=-1)
    if need_cast:
        boxes.round_()
        boxes = boxes.to(dtype)
    return boxes


def _box_xyxyxyxy_to_xywhr(boxes: Tensor) -> Tensor:
    """
    Converts rotated bounding boxes from (x1, y1, x2, y2, x3, y3, x4, y4) format to (x1, y1, w, h, r) format.
    (x1, y1) refer to top left of the rotated bounding box
    (x2, y2) refer to bottom left of the rotated bounding box
    (x3, y3) refer to bottom right of the rotated bounding box
    (x4, y4) refer to top right of the rotated bounding box
    (w, h) refers to width and height of rotated bounding box
    r is rotation angle w.r.t to the box center by :math:`|r|` degrees counter clock wise in the image plan

    Args:
        boxes (Tensor(N, 8)): rotated boxes in (x1, y1, x2, y2, x3, y3, x4, y4) format.

    Returns:
        boxes (Tensor[N, 5]): rotated boxes in (x1, y1, w, h, r) format.
    """
    dtype = boxes.dtype
    need_cast = not boxes.is_floating_point()
    x1, y1, x2, y2, x3, y3, x4, y4 = boxes.unbind(-1)
    r_rad = torch.atan2(y1 - y2, x2 - x1)
    r = r_rad * 180 / torch.pi

    w = ((x2 - x1) ** 2 + (y1 - y2) ** 2).sqrt()
    h = ((x3 - x2) ** 2 + (y3 - y2) ** 2).sqrt()

    boxes = torch.stack((x1, y1, w, h, r), dim=-1)
    if need_cast:
        boxes.round_()
        boxes = boxes.to(dtype)
    return boxes
