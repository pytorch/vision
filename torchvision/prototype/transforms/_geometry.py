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
from torchvision.transforms.functional_tensor import _parse_pad_padding
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

        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
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

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.Image):
            output = F.five_crop_image_tensor(inpt, self.size)
            return MultiCropResult(features.Image.new_like(inpt, o) for o in output)
        elif is_simple_tensor(inpt):
            return MultiCropResult(F.five_crop_image_tensor(inpt, self.size))
        elif isinstance(inpt, PIL.Image.Image):
            return MultiCropResult(F.five_crop_image_pil(inpt, self.size))
        else:
            return inpt

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

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, features.Image):
            output = F.ten_crop_image_tensor(inpt, self.size, vertical_flip=self.vertical_flip)
            return MultiCropResult(features.Image.new_like(inpt, o) for o in output)
        elif is_simple_tensor(inpt):
            return MultiCropResult(F.ten_crop_image_tensor(inpt, self.size))
        elif isinstance(inpt, PIL.Image.Image):
            return MultiCropResult(F.ten_crop_image_pil(inpt, self.size))
        else:
            return inpt

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


def _check_padding_arg(padding: Union[int, Sequence[int]]) -> None:
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")

    if isinstance(padding, (tuple, list)) and len(padding) not in [1, 2, 4]:
        raise ValueError(f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple")


# TODO: let's use torchvision._utils.StrEnum to have the best of both worlds (strings and enums)
# https://github.com/pytorch/vision/issues/6250
def _check_padding_mode_arg(padding_mode: Literal["constant", "edge", "reflect", "symmetric"]) -> None:
    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")


class Pad(Transform):
    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ) -> None:
        super().__init__()

        _check_padding_arg(padding)
        _check_fill_arg(fill)
        _check_padding_mode_arg(padding_mode)

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.pad(inpt, padding=self.padding, fill=self.fill, padding_mode=self.padding_mode)


class RandomZoomOut(_RandomApplyTransform):
    def __init__(
        self,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        side_range: Sequence[float] = (1.0, 4.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        _check_fill_arg(fill)
        self.fill = fill

        _check_sequence_input(side_range, "side_range", req_sizes=(2,))

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

        return dict(padding=padding, fill=self.fill)

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


class RandomCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        padding: Optional[Union[int, Sequence[int]]] = None,
        pad_if_needed: bool = False,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ) -> None:
        super().__init__()

        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if pad_if_needed or padding is not None:
            if padding is not None:
                _check_padding_arg(padding)
            _check_fill_arg(fill)
            _check_padding_mode_arg(padding_mode)

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image = query_image(sample)
        _, height, width = get_image_dimensions(image)

        if self.padding is not None:
            # update height, width with static padding data
            padding = self.padding
            if isinstance(padding, Sequence):
                padding = list(padding)
            pad_left, pad_right, pad_top, pad_bottom = _parse_pad_padding(padding)
            height += pad_top + pad_bottom
            width += pad_left + pad_right

        output_height, output_width = self.size
        # We have to store maybe padded image size for pad_if_needed branch in _transform
        input_height, input_width = height, width

        if self.pad_if_needed:
            # pad width if needed
            if width < output_width:
                width += 2 * (output_width - width)
            # pad height if needed
            if height < output_height:
                height += 2 * (output_height - height)

        if height + 1 < output_height or width + 1 < output_width:
            raise ValueError(
                f"Required crop size {(output_height, output_width)} is larger then input image size {(height, width)}"
            )

        if width == output_width and height == output_height:
            return dict(top=0, left=0, height=height, width=width, input_width=input_width, input_height=input_height)

        top = torch.randint(0, height - output_height + 1, size=(1,)).item()
        left = torch.randint(0, width - output_width + 1, size=(1,)).item()

        return dict(
            top=top,
            left=left,
            height=output_height,
            width=output_width,
            input_width=input_width,
            input_height=input_height,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if self.padding is not None:
            inpt = F.pad(inpt, padding=self.padding, fill=self.fill, padding_mode=self.padding_mode)

        if self.pad_if_needed:
            input_width, input_height = params["input_width"], params["input_height"]
            if input_width < self.size[1]:
                padding = [self.size[1] - input_width, 0]
                inpt = F.pad(inpt, padding=padding, fill=self.fill, padding_mode=self.padding_mode)
            if input_height < self.size[0]:
                padding = [0, self.size[0] - input_height]
                inpt = F.pad(inpt, padding=padding, fill=self.fill, padding_mode=self.padding_mode)

        return F.crop(inpt, top=params["top"], left=params["left"], height=params["height"], width=params["width"])


class RandomPerspective(_RandomApplyTransform):
    def __init__(
        self,
        distortion_scale: float,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        _check_fill_arg(fill)
        if not (0 <= distortion_scale <= 1):
            raise ValueError("Argument distortion_scale value should be between 0 and 1")

        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.fill = fill

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        # Get image size
        # TODO: make it work with bboxes and segm masks
        image = query_image(sample)
        _, height, width = get_image_dimensions(image)

        distortion_scale = self.distortion_scale

        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return dict(startpoints=startpoints, endpoints=endpoints)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.perspective(
            inpt,
            **params,
            fill=self.fill,
            interpolation=self.interpolation,
        )


def _setup_float_or_seq(arg: Union[float, Sequence[float]], name: str, req_size: int = 2) -> Sequence[float]:
    if not isinstance(arg, (float, Sequence)):
        raise TypeError(f"{name} should be float or a sequence of floats. Got {type(arg)}")
    if isinstance(arg, Sequence) and len(arg) != req_size:
        raise ValueError(f"If {name} is a sequence its length should be one of {req_size}. Got {len(arg)}")
    if isinstance(arg, Sequence):
        for element in arg:
            if not isinstance(element, float):
                raise ValueError(f"{name} should be a sequence of floats. Got {type(element)}")

    if isinstance(arg, float):
        arg = [float(arg), float(arg)]
    if isinstance(arg, (list, tuple)) and len(arg) == 1:
        arg = [arg[0], arg[0]]
    return arg


class ElasticTransform(Transform):
    def __init__(
        self,
        alpha: Union[float, Sequence[float]] = 50.0,
        sigma: Union[float, Sequence[float]] = 5.0,
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.alpha = _setup_float_or_seq(alpha, "alpha", 2)
        self.sigma = _setup_float_or_seq(sigma, "sigma", 2)

        _check_fill_arg(fill)

        self.interpolation = interpolation
        self.fill = fill

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        # Get image size
        # TODO: make it work with bboxes and segm masks
        image = query_image(sample)
        _, *size = get_image_dimensions(image)

        dx = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma[0] > 0.0:
            kx = int(8 * self.sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = F.gaussian_blur(dx, [kx, kx], list(self.sigma))
        dx = dx * self.alpha[0] / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma[1] > 0.0:
            ky = int(8 * self.sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = F.gaussian_blur(dy, [ky, ky], list(self.sigma))
        dy = dy * self.alpha[1] / size[1]
        displacement = torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2
        return dict(displacement=displacement)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.elastic(
            inpt,
            **params,
            fill=self.fill,
            interpolation=self.interpolation,
        )


class ScaleJitter(Transform):
    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image = query_image(sample)
        _, orig_height, orig_width = get_image_dimensions(image)

        r = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        new_width = int(self.target_size[1] * r)
        new_height = int(self.target_size[0] * r)

        return dict(size=(new_height, new_width))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(inpt, size=params["size"], interpolation=self.interpolation)
