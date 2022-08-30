import math
import numbers
import warnings
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torchvision.ops.boxes import box_iou
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, InterpolationMode, Transform

from typing_extensions import Literal

from ._transform import _RandomApplyTransform
from ._utils import (
    _check_sequence_input,
    _parse_pad_padding,
    _setup_angle,
    _setup_size,
    has_all,
    has_any,
    query_bounding_box,
    query_chw,
)


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
        _, height, width = query_chw(sample)
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


class FiveCrop(Transform):
    """
    Example:
        >>> class BatchMultiCrop(transforms.Transform):
        ...     def forward(self, sample: Tuple[Tuple[features.Image, ...], features.Label]):
        ...         images, labels = sample
        ...         batch_size = len(images)
        ...         images = features.Image.new_like(images[0], torch.stack(images))
        ...         labels = features.Label.new_like(labels, labels.repeat(batch_size))
        ...         return images, labels
        ...
        >>> image = features.Image(torch.rand(3, 256, 256))
        >>> label = features.Label(0)
        >>> transform = transforms.Compose([transforms.FiveCrop(), BatchMultiCrop()])
        >>> images, labels = transform(image, label)
        >>> images.shape
        torch.Size([5, 3, 224, 224])
        >>> labels.shape
        torch.Size([5])
    """

    _transformed_types = (features.Image, PIL.Image.Image, features.is_simple_tensor)

    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.five_crop(inpt, self.size)

    def forward(self, *inputs: Any) -> Any:
        if has_any(inputs, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(*inputs)


class TenCrop(Transform):
    """
    See :class:`~torchvision.prototype.transforms.FiveCrop` for an example.
    """

    _transformed_types = (features.Image, PIL.Image.Image, features.is_simple_tensor)

    def __init__(self, size: Union[int, Sequence[int]], vertical_flip: bool = False) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.vertical_flip = vertical_flip

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.ten_crop(inpt, self.size, vertical_flip=self.vertical_flip)

    def forward(self, *inputs: Any) -> Any:
        if has_any(inputs, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        return super().forward(*inputs)


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
        orig_c, orig_h, orig_w = query_chw(sample)

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
        _, height, width = query_chw(sample)

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
        _, height, width = query_chw(sample)

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
        # TODO: (PERF) check for speed optimization if we avoid repeated pad calls
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
        _, height, width = query_chw(sample)

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
        _, *size = query_chw(sample)

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


class RandomIoUCrop(Transform):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        _, orig_h, orig_w = query_chw(sample)
        bboxes = query_bounding_box(sample)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return dict()

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                xyxy_bboxes = F.convert_bounding_box_format(
                    bboxes, old_format=bboxes.format, new_format=features.BoundingBoxFormat.XYXY, copy=True
                )
                cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(
                    xyxy_bboxes,
                    torch.tensor([[left, top, right, bottom]], dtype=xyxy_bboxes.dtype, device=xyxy_bboxes.device),
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                return dict(top=top, left=left, height=new_h, width=new_w, is_within_crop_area=is_within_crop_area)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt

        is_within_crop_area = params["is_within_crop_area"]

        if isinstance(inpt, (features.Label, features.OneHotLabel)):
            return inpt.new_like(inpt, inpt[is_within_crop_area])  # type: ignore[arg-type]

        output = F.crop(inpt, top=params["top"], left=params["left"], height=params["height"], width=params["width"])

        if isinstance(output, features.BoundingBox):
            bboxes = output[is_within_crop_area]
            bboxes = F.clamp_bounding_box(bboxes, output.format, output.image_size)
            output = features.BoundingBox.new_like(output, bboxes)
        elif isinstance(output, features.SegmentationMask) and output.shape[-3] > 1:
            # apply is_within_crop_area if mask is one-hot encoded
            masks = output[is_within_crop_area]
            output = features.SegmentationMask.new_like(output, masks)

        return output

    def forward(self, *inputs: Any) -> Any:
        if not (
            has_all(inputs, features.BoundingBox)
            and has_any(inputs, PIL.Image.Image, features.Image, features.is_simple_tensor)
            and has_any(inputs, features.Label, features.OneHotLabel)
        ):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain Images or PIL Images, "
                "BoundingBoxes and Labels or OneHotLabels. Sample can also contain Segmentation Masks."
            )
        return super().forward(*inputs)


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
        _, orig_height, orig_width = query_chw(sample)

        r = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        new_width = int(self.target_size[1] * r)
        new_height = int(self.target_size[0] * r)

        return dict(size=(new_height, new_width))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(inpt, size=params["size"], interpolation=self.interpolation)


class RandomShortestSize(Transform):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        _, orig_height, orig_width = query_chw(sample)

        min_size = self.min_size[int(torch.randint(len(self.min_size), ()))]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        return dict(size=(new_height, new_width))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(inpt, size=params["size"], interpolation=self.interpolation)


class FixedSizeCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        fill: Union[int, float, Sequence[int], Sequence[float]] = 0,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__()
        size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        _, height, width = query_chw(sample)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        needs_crop = new_height != height or new_width != width

        offset_height = max(height - self.crop_height, 0)
        offset_width = max(width - self.crop_width, 0)

        r = torch.rand(1)
        top = int(offset_height * r)
        left = int(offset_width * r)

        try:
            bounding_boxes = query_bounding_box(sample)
        except ValueError:
            bounding_boxes = None

        if needs_crop and bounding_boxes is not None:
            bounding_boxes = cast(
                features.BoundingBox, F.crop(bounding_boxes, top=top, left=left, height=height, width=width)
            )
            bounding_boxes = features.BoundingBox.new_like(
                bounding_boxes,
                F.clamp_bounding_box(
                    bounding_boxes, format=bounding_boxes.format, image_size=bounding_boxes.image_size
                ),
            )
            height_and_width = bounding_boxes.to_format(features.BoundingBoxFormat.XYWH)[..., 2:]
            is_valid = torch.all(height_and_width > 0, dim=-1)
        else:
            is_valid = None

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)

        needs_pad = pad_bottom != 0 or pad_right != 0

        return dict(
            needs_crop=needs_crop,
            top=top,
            left=left,
            height=new_height,
            width=new_width,
            is_valid=is_valid,
            padding=[0, 0, pad_right, pad_bottom],
            needs_pad=needs_pad,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["needs_crop"]:
            inpt = F.crop(
                inpt,
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )

        if params["is_valid"] is not None:
            if isinstance(inpt, (features.Label, features.OneHotLabel, features.SegmentationMask)):
                inpt = inpt.new_like(inpt, inpt[params["is_valid"]])  # type: ignore[arg-type]
            elif isinstance(inpt, features.BoundingBox):
                inpt = features.BoundingBox.new_like(
                    inpt,
                    F.clamp_bounding_box(inpt[params["is_valid"]], format=inpt.format, image_size=inpt.image_size),
                )

        if params["needs_pad"]:
            inpt = F.pad(inpt, params["padding"], fill=self.fill, padding_mode=self.padding_mode)

        return inpt

    def forward(self, *inputs: Any) -> Any:
        if not has_any(inputs, PIL.Image.Image, features.Image, features.is_simple_tensor):
            raise TypeError(f"{type(self).__name__}() requires input sample to contain an tensor or PIL image.")

        if has_any(inputs, features.BoundingBox) and not has_any(inputs, features.Label, features.OneHotLabel):
            raise TypeError(
                f"If a BoundingBox is contained in the input sample, "
                f"{type(self).__name__}() also requires it to contain a Label or OneHotLabel."
            )

        return super().forward(*inputs)
