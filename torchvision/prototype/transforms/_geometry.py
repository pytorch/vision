import collections.abc
import math
import numbers
import warnings
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size
from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from ._utils import get_image_dimensions, has_any, is_simple_tensor, query_image


class RandomHorizontalFlip(_RandomApplyTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.horizontal_flip(inpt)


class RandomVerticalFlip(_RandomApplyTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.vertical_flip(inpt)


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.size = [size] if isinstance(size, int) else list(size)
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(
            inpt,
            self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )


class CenterCrop(Transform):
    def __init__(self, output_size: List[int]):
        super().__init__()
        self.output_size = output_size

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.center_crop(inpt, output_size=self.output_size)


class RandomResizedCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        scale = cast(Tuple[float, float], scale)
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        ratio = cast(Tuple[float, float], ratio)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        # vfdev-5: techically, this op can work on bboxes/segm masks only inputs without image in samples
        # What if we have multiple images/bboxes/masks of different sizes ?
        # TODO: let's support bbox or mask in samples without image
        image = query_image(sample)
        _, height, width = get_image_dimensions(image)
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resized_crop(
            inpt, **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
        )


class MultiCropResult(list):
    """Helper class for :class:`~torchvision.prototype.transforms.BatchMultiCrop`.

    Outputs of multi crop transforms such as :class:`~torchvision.prototype.transforms.FiveCrop` and
    `:class:`~torchvision.prototype.transforms.TenCrop` should be wrapped in this in order to be batched correctly by
    :class:`~torchvision.prototype.transforms.BatchMultiCrop`.
    """

    pass


class FiveCrop(Transform):
    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.five_crop_image_tensor(input, self.size)
            return MultiCropResult(features.Image.new_like(input, o) for o in output)
        elif is_simple_tensor(input):
            return MultiCropResult(F.five_crop_image_tensor(input, self.size))
        elif isinstance(input, PIL.Image.Image):
            return MultiCropResult(F.five_crop_image_pil(input, self.size))
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class TenCrop(Transform):
    def __init__(self, size: Union[int, Sequence[int]], vertical_flip: bool = False) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.vertical_flip = vertical_flip

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.ten_crop_image_tensor(input, self.size, vertical_flip=self.vertical_flip)
            return MultiCropResult(features.Image.new_like(input, o) for o in output)
        elif is_simple_tensor(input):
            return MultiCropResult(F.ten_crop_image_tensor(input, self.size))
        elif isinstance(input, PIL.Image.Image):
            return MultiCropResult(F.ten_crop_image_pil(input, self.size))
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(sample)


class BatchMultiCrop(Transform):
    def forward(self, *inputs: Any) -> Any:
        # This is basically the functionality of `torchvision.prototype.utils._internal.apply_recursively` with one
        # significant difference:
        # Since we need multiple images to batch them together, we need to explicitly exclude `MultiCropResult` from
        # the sequence case.
        def apply_recursively(obj: Any) -> Any:
            if isinstance(obj, MultiCropResult):
                crops = obj
                if isinstance(obj[0], PIL.Image.Image):
                    crops = [pil_to_tensor(crop) for crop in crops]  # type: ignore[assignment]

                batch = torch.stack(crops)

                if isinstance(obj[0], features.Image):
                    batch = features.Image.new_like(obj[0], batch)

                return batch
            elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
                return [apply_recursively(item) for item in obj]
            elif isinstance(obj, collections.abc.Mapping):
                return {key: apply_recursively(item) for key, item in obj.items()}
            else:
                return obj

        return apply_recursively(inputs if len(inputs) > 1 else inputs[0])


def _check_fill_arg(fill: Union[int, float, Sequence[int], Sequence[float]]) -> None:
    if not isinstance(fill, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate fill arg")


class Pad(Transform):
    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ) -> None:
        super().__init__()
        if not isinstance(padding, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate padding arg")

        _check_fill_arg(fill)

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError(
                f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple"
            )

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.pad(inpt, padding=self.padding, fill=self.fill, padding_mode=self.padding_mode)


class RandomZoomOut(_RandomApplyTransform):
    def __init__(
        self,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        side_range: Tuple[float, float] = (1.0, 4.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        _check_fill_arg(fill)
        self.fill = fill

        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image = query_image(sample)
        orig_c, orig_h, orig_w = get_image_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        padding = [left, top, right, bottom]

        fill = self.fill
        if not isinstance(fill, collections.abc.Sequence):
            fill = [fill] * orig_c

        return dict(padding=padding, fill=fill)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.pad(inpt, **params)


class RandomRotation(Transform):
    def __init__(
        self,
        degrees: Union[numbers.Number, Sequence],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        center: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.interpolation = interpolation
        self.expand = expand

        _check_fill_arg(fill)

        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        return dict(angle=angle)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.rotate(
            inpt,
            **params,
            interpolation=self.interpolation,
            expand=self.expand,
            fill=self.fill,
            center=self.center,
        )


class RandomAffine(Transform):
    def __init__(
        self,
        degrees: Union[numbers.Number, Sequence],
        translate: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        shear: Optional[Union[float, Sequence[float]]] = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        center: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation

        _check_fill_arg(fill)

        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    def _get_params(self, sample: Any) -> Dict[str, Any]:

        # Get image size
        # TODO: make it work with bboxes and segm masks
        image = query_image(sample)
        _, height, width = get_image_dimensions(image)

        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        if self.translate is not None:
            max_dx = float(self.translate[0] * width)
            max_dy = float(self.translate[1] * height)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if self.scale is not None:
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if self.shear is not None:
            shear_x = float(torch.empty(1).uniform_(self.shear[0], self.shear[1]).item())
            if len(self.shear) == 4:
                shear_y = float(torch.empty(1).uniform_(self.shear[2], self.shear[3]).item())

        shear = (shear_x, shear_y)
        return dict(angle=angle, translations=translations, scale=scale, shear=shear)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.affine(
            inpt,
            **params,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center,
        )
