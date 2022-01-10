import torch
from torchvision.prototype.features import BoundingBoxFormat


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xyxy = xywh.clone()
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


def convert_format_bounding_box(
    bounding_box: torch.Tensor,
    *,
    old_format: BoundingBoxFormat,
    new_format: BoundingBoxFormat,
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


def _max_value(dtype: torch.dtype) -> float:
    # TODO: replace this method with torch.iinfo when it gets torchscript support.
    # https://github.com/pytorch/pytorch/issues/41492

    a = torch.tensor(2, dtype=dtype)
    signed = 1 if torch.tensor(0, dtype=dtype).is_signed() else 0
    bits = 1
    max_value = torch.tensor(-signed, dtype=torch.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            break
    return max_value.item()


def convert_dtype_image(image: torch.Tensor, *, new_dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == new_dtype:
        return image

    if image.is_floating_point():

        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=new_dtype).is_floating_point():
            return image.to(new_dtype)

        # float to int
        if (image.dtype == torch.float32 and new_dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and new_dtype == torch.int64
        ):
            raise RuntimeError(f"The cast from {image.dtype} to {new_dtype} cannot be performed safely.")

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = _max_value(new_dtype)
        result = image.mul(max_val + 1.0 - eps)
        return result.to(new_dtype)
    else:
        input_max = _max_value(image.dtype)

        # int to float
        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=new_dtype).is_floating_point():
            image = image.to(new_dtype)
            return image / input_max

        output_max = _max_value(new_dtype)

        # int to int
        if input_max > output_max:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image // factor can produce different results
            factor = int((input_max + 1) // (output_max + 1))
            image = torch.div(image, factor, rounding_mode="floor")
            return image.to(new_dtype)
        else:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image * factor can produce different results
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(new_dtype)
            return image * factor
