import enum
import functools
from typing import Callable, Union, Tuple, Dict, Any, Optional, cast

import torch
from torchvision.prototype.utils._internal import StrEnum

from ._feature import Feature, DEFAULT


class BoundingBoxFormat(StrEnum):
    # this is just for test purposes
    _SENTINEL = -1
    XYXY = enum.auto()
    XYWH = enum.auto()
    CXCYWH = enum.auto()


def to_parts(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return input.unbind(dim=-1)  # type: ignore[return-value]


def from_parts(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.stack((a, b, c, d), dim=-1)


def format_converter_wrapper(
    part_converter: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    def wrapper(input: torch.Tensor) -> torch.Tensor:
        return from_parts(*part_converter(*to_parts(input)))

    return wrapper


@format_converter_wrapper
def xywh_to_xyxy(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2


@format_converter_wrapper
def xyxy_to_xywh(
    x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


@format_converter_wrapper
def cxcywh_to_xyxy(
    cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return x1, y1, x2, y2


@format_converter_wrapper
def xyxy_to_cxcywh(
    x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


class BoundingBox(Feature):
    formats = BoundingBoxFormat
    format: BoundingBoxFormat
    image_size: Tuple[int, int]

    @classmethod
    def _parse_meta_data(
        cls,
        format: Union[str, BoundingBoxFormat] = DEFAULT,  # type: ignore[assignment]
        image_size: Optional[Tuple[int, int]] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        if isinstance(format, str):
            format = BoundingBoxFormat[format]
        format_fallback = BoundingBoxFormat.XYXY
        return dict(
            format=(format, format_fallback),
            image_size=(image_size, functools.partial(cls.guess_image_size, format=format_fallback)),
        )

    _TO_XYXY_MAP = {
        BoundingBoxFormat.XYWH: xywh_to_xyxy,
        BoundingBoxFormat.CXCYWH: cxcywh_to_xyxy,
    }
    _FROM_XYXY_MAP = {
        BoundingBoxFormat.XYWH: xyxy_to_xywh,
        BoundingBoxFormat.CXCYWH: xyxy_to_cxcywh,
    }

    @classmethod
    def guess_image_size(cls, data: torch.Tensor, *, format: BoundingBoxFormat) -> Tuple[int, int]:
        if format not in (BoundingBoxFormat.XYWH, BoundingBoxFormat.CXCYWH):
            if format != BoundingBoxFormat.XYXY:
                data = cls._TO_XYXY_MAP[format](data)
            data = cls._FROM_XYXY_MAP[BoundingBoxFormat.XYWH](data)
        *_, w, h = to_parts(data)
        if data.dtype.is_floating_point:
            w = w.ceil()
            h = h.ceil()
        return int(h.max()), int(w.max())

    @classmethod
    def from_parts(
        cls,
        a,
        b,
        c,
        d,
        *,
        like: Optional["BoundingBox"] = None,
        format: Union[str, BoundingBoxFormat] = DEFAULT,  # type: ignore[assignment]
        image_size: Optional[Tuple[int, int]] = DEFAULT,  # type: ignore[assignment]
    ) -> "BoundingBox":
        return cls(from_parts(a, b, c, d), like=like, image_size=image_size, format=format)

    def to_parts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return to_parts(self)

    def convert(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        if format == self.format:
            return cast(BoundingBox, self.clone())

        data = self

        if self.format != BoundingBoxFormat.XYXY:
            data = self._TO_XYXY_MAP[self.format](data)

        if format != BoundingBoxFormat.XYXY:
            data = self._FROM_XYXY_MAP[format](data)

        return BoundingBox(data, like=self, format=format)
