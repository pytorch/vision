from typing import Any, Dict, Tuple, Union

import torch
from torch.nn.functional import interpolate
from torchvision.prototype.datasets.utils import SampleQuery
from torchvision.prototype.features import BoundingBox, Image, Label
from torchvision.prototype.transforms import Transform


class HorizontalFlip(Transform):
    NO_OP_FEATURE_TYPES = {Label}

    @staticmethod
    def image(input: Image) -> Image:
        return Image(input.flip((-1,)), like=input)

    @staticmethod
    def bounding_box(input: BoundingBox) -> BoundingBox:
        x, y, w, h = input.convert("xywh").to_parts()
        x = input.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, like=input, format="xywh").convert(input.format)


class Resize(Transform):
    NO_OP_FEATURE_TYPES = {Label}

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        *,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation_mode = interpolation_mode

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(size=self.size, interpolation_mode=self.interpolation_mode)

    @staticmethod
    def image(input: Image, *, size: Tuple[int, int], interpolation_mode: str = "nearest") -> Image:
        return Image(interpolate(input.unsqueeze(0), size, mode=interpolation_mode).squeeze(0), like=input)

    @staticmethod
    def bounding_box(input: BoundingBox, *, size: Tuple[int, int], **_: Any) -> BoundingBox:
        old_height, old_width = input.image_size
        new_height, new_width = size

        height_scale = new_height / old_height
        width_scale = new_width / old_width

        old_x1, old_y1, old_x2, old_y2 = input.convert("xyxy").to_parts()

        new_x1 = old_x1 * width_scale
        new_y1 = old_y1 * height_scale

        new_x2 = old_x2 * width_scale
        new_y2 = old_y2 * height_scale

        return BoundingBox.from_parts(
            new_x1, new_y1, new_x2, new_y2, like=input, format="xyxy", image_size=size
        ).convert(input.format)

    def extra_repr(self) -> str:
        extra_repr = f"size={self.size}"
        if self.interpolation_mode != "bilinear":
            extra_repr += f", interpolation_mode={self.interpolation_mode}"
        return extra_repr


class RandomResize(Transform, wraps=Resize):
    def __init__(self, min_size: Union[int, Tuple[int, int]], max_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.min_size = (min_size, min_size) if isinstance(min_size, int) else min_size
        self.max_size = (max_size, max_size) if isinstance(max_size, int) else max_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        min_height, min_width = self.min_size
        max_height, max_width = self.max_size
        height = int(torch.randint(min_height, max_height + 1, size=()))
        width = int(torch.randint(min_width, max_width + 1, size=()))
        return dict(size=(height, width))

    def extra_repr(self) -> str:
        return f"min_size={self.min_size}, max_size={self.max_size}"


class Crop(Transform):
    NO_OP_FEATURE_TYPES = {BoundingBox, Label}

    def __init__(self, crop_box: BoundingBox) -> None:
        super().__init__()
        self.crop_box = crop_box.convert("xyxy")

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(crop_box=self.crop_box)

    @staticmethod
    def image(input: Image, *, crop_box: BoundingBox) -> Image:
        # FIXME: pad input in case it is smaller than crop_box
        x1, y1, x2, y2 = crop_box.convert("xyxy").to_parts()
        return Image(input[..., y1 : y2 + 1, x1 : x2 + 1], like=input)  # type: ignore[misc]


class CenterCrop(Transform, wraps=Crop):
    def __init__(self, crop_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = SampleQuery(sample).image_size()
        image_height, image_width = image_size
        cx = image_width // 2
        cy = image_height // 2
        h, w = self.crop_size
        crop_box = BoundingBox.from_parts(cx, cy, w, h, image_size=image_size, format="cxcywh")
        return dict(crop_box=crop_box)

    def extra_repr(self) -> str:
        return f"crop_size={self.crop_size}"


class RandomCrop(Transform, wraps=Crop):
    def __init__(self, crop_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = SampleQuery(sample).image_size()
        image_height, image_width = image_size
        crop_height, crop_width = self.crop_size
        x = torch.randint(0, image_width - crop_width + 1, size=()) if crop_width < image_width else 0
        y = torch.randint(0, image_height - crop_height + 1, size=()) if crop_height < image_height else 0
        crop_box = BoundingBox.from_parts(x, y, crop_width, crop_height, image_size=image_size, format="xywh")
        return dict(crop_box=crop_box)

    def extra_repr(self) -> str:
        return f"crop_size={self.crop_size}"
