import math
import numbers
import warnings
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import PIL.Image

import torch
from torchvision.ops import masks_to_boxes
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F, InterpolationMode

from ._transform import _DetectionBatchTransform, _RandomApplyTransform
from .utils import has_any, is_simple_tensor, query_chw, query_spatial_size


D = TypeVar("D", bound=datapoints._datapoint.Datapoint)


class RandomErasing(_RandomApplyTransform):
    _transformed_types = (is_simple_tensor, datapoints.Image, PIL.Image.Image, datapoints.Video)

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        self.scale = scale
        self.ratio = ratio
        if isinstance(value, (int, float)):
            self.value = [value]
        elif isinstance(value, str):
            self.value = None
        elif isinstance(value, tuple):
            self.value = list(value)
        else:
            self.value = value
        self.inplace = inplace

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_c, img_h, img_w = query_chw(flat_inputs)

        if self.value is not None and not (len(self.value) in (1, img_c)):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of inpt channels)"
            )

        area = img_h * img_w

        log_ratio = self._log_ratio
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if self.value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(self.value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break
        else:
            i, j, h, w, v = 0, 0, img_h, img_w, None

        return dict(i=i, j=j, h=h, w=w, v=v)

    def _transform(
        self, inpt: Union[datapoints.ImageType, datapoints.VideoType], params: Dict[str, Any]
    ) -> Union[datapoints.ImageType, datapoints.VideoType]:
        if params["v"] is not None:
            inpt = F.erase(inpt, **params, inplace=self.inplace)

        return inpt


class _BaseMixupCutmix(_RandomApplyTransform):
    def __init__(self, alpha: float, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if not (
            has_any(flat_inputs, datapoints.Image, datapoints.Video, is_simple_tensor)
            and has_any(flat_inputs, datapoints.OneHotLabel)
        ):
            raise TypeError(f"{type(self).__name__}() is only defined for tensor images/videos and one-hot labels.")
        if has_any(flat_inputs, PIL.Image.Image, datapoints.BoundingBox, datapoints.Mask, datapoints.Label):
            raise TypeError(
                f"{type(self).__name__}() does not support PIL images, bounding boxes, masks and plain labels."
            )

    def _mixup_onehotlabel(self, inpt: datapoints.OneHotLabel, lam: float) -> datapoints.OneHotLabel:
        if inpt.ndim < 2:
            raise ValueError("Need a batch of one hot labels")
        output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))
        return datapoints.OneHotLabel.wrap_like(inpt, output)


class RandomMixup(_BaseMixupCutmix):
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]
        if isinstance(inpt, (datapoints.Image, datapoints.Video)) or is_simple_tensor(inpt):
            expected_ndim = 5 if isinstance(inpt, datapoints.Video) else 4
            if inpt.ndim < expected_ndim:
                raise ValueError("The transform expects a batched input")
            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(inpt, (datapoints.Image, datapoints.Video)):
                output = type(inpt).wrap_like(inpt, output)  # type: ignore[arg-type]

            return output
        elif isinstance(inpt, datapoints.OneHotLabel):
            return self._mixup_onehotlabel(inpt, lam)
        else:
            return inpt


class RandomCutmix(_BaseMixupCutmix):
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_spatial_size(flat_inputs)

        r_x = torch.randint(W, ())
        r_y = torch.randint(H, ())

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, (datapoints.Image, datapoints.Video)) or is_simple_tensor(inpt):
            box = params["box"]
            expected_ndim = 5 if isinstance(inpt, datapoints.Video) else 4
            if inpt.ndim < expected_ndim:
                raise ValueError("The transform expects a batched input")
            x1, y1, x2, y2 = box
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

            if isinstance(inpt, (datapoints.Image, datapoints.Video)):
                output = inpt.wrap_like(inpt, output)  # type: ignore[arg-type]

            return output
        elif isinstance(inpt, datapoints.OneHotLabel):
            lam_adjusted = params["lam_adjusted"]
            return self._mixup_onehotlabel(inpt, lam_adjusted)
        else:
            return inpt


class SimpleCopyPaste(_DetectionBatchTransform):
    def __init__(
        self,
        blending: bool = True,
        resize_interpolation: InterpolationMode = F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending
        self.antialias = antialias

    def forward(self, *inputs: Any) -> Any:
        flat_batch_with_spec, batch = self._flatten_and_extract_data(
            inputs,
            image=(datapoints.Image, PIL.Image.Image, is_simple_tensor),
            boxes=(datapoints.BoundingBox,),
            masks=(datapoints.Mask,),
            labels=(datapoints.Label, datapoints.OneHotLabel),
        )
        batch = self._to_image_tensor(batch)

        batch_output = []
        for sample, sample_rolled in zip(batch, batch[-1:] + batch[:-1]):
            num_masks = len(sample_rolled["masks"])
            if num_masks < 1:
                # This might for example happen with the LSJ augmentation strategy
                batch_output.append(sample)
                continue

            random_selection = torch.randint(0, num_masks, (num_masks,), device=sample_rolled["masks"].device)
            random_selection = torch.unique(random_selection)

            batch_output.append(
                self._simple_copy_paste(
                    sample,
                    sample_rolled,
                    random_selection=random_selection,
                    blending=self.blending,
                    resize_interpolation=self.resize_interpolation,
                    antialias=self.antialias,
                )
            )

        return self._unflatten_and_insert_data(flat_batch_with_spec, batch_output)

    @staticmethod
    def _wrapping_getitem(datapoint: D, index: Any) -> D:
        return type(datapoint).wrap_like(datapoint, datapoint[index])

    def _simple_copy_paste(
        self,
        sample_1: Dict[str, Any],
        sample_2: Dict[str, Any],
        *,
        random_selection: torch.Tensor,
        blending: bool,
        resize_interpolation: F.InterpolationMode,
        antialias: Optional[bool],
    ) -> Dict[str, Any]:
        dst_image = sample_1["image"]
        dst_masks = sample_1["masks"]
        dst_labels = sample_1["labels"]

        src_image = sample_2["image"]
        src_masks = self._wrapping_getitem(sample_2["masks"], random_selection)
        src_boxes = self._wrapping_getitem(sample_2["dst_boxes"], random_selection)
        src_labels = self._wrapping_getitem(sample_2["labels"], random_selection)

        # In case the `dst_image` and `src_image` have different spatial sizes, we resize `src_image` and the
        # corresponding annotations to `dst_image`'s spatial size. This differs from the official implementation, since
        # that only works with equally sized data, e.g. coming from the LSJ augmentation strategy.
        dst_spatial_size = dst_image.shape[-2:]
        src_spatial_size = src_image.shape[-2:]
        if dst_spatial_size != src_spatial_size:
            src_image = F.resize(
                src_image, size=dst_spatial_size, interpolation=resize_interpolation, antialias=antialias
            )
            src_masks = F.resize(src_masks, size=dst_spatial_size)
            src_boxes = F.resize(src_boxes, size=dst_spatial_size)

        src_paste_mask = src_masks.sum(dim=0, keepdim=0) > 0
        # Although the parameter is called "blending", we don't actually blend here. `src_paste_mask` is a boolean
        # mask and although `F.gaussian_blur` internally converts to floating point, it will be converted back to
        # boolean on the way out. Meaning, although we blur, `src_paste_mask` will have no values other than 0 or 1.
        # The original paper doesn't specify how blending should be done and the official implementation is not helpful
        # either:
        # https://github.com/tensorflow/tpu/blob/732902a457b2a8924f885ee832830e1bf6d7c537/models/official/detection/dataloader/maskrcnn_parser_with_copy_paste.py#L331-L334
        if blending:
            src_paste_mask = F.gaussian_blur(src_paste_mask, kernel_size=[5, 5], sigma=[2.0])
        dst_paste_mask = src_paste_mask.logical_not()

        image = datapoints.Image.wrap_like(dst_image, dst_image.mul(dst_paste_mask).add_(src_image.mul(src_paste_mask)))

        dst_masks = dst_masks * dst_paste_mask
        # Since we paste the `src_image` into the `dst_image`, we might completely cover an object previously visible in
        # `dst_image`. Furthermore, with `blending=True` small regions to begin with might also be shrunk enough to
        # vanish. Thus, we check for degenerate masks and remove them.
        valid_dst_masks = dst_masks.sum((-1, -2)) > 0
        dst_masks = dst_masks[valid_dst_masks]
        masks = datapoints.Mask.wrap_like(dst_masks, torch.cat([dst_masks, src_masks]))

        # Since the `dst_masks` might have changed above, we recompute the corresponding `dst_boxes`.
        dst_boxes_xyxy = masks_to_boxes(dst_masks)
        # `masks_to_boxes` produces boxes with x2y2 inclusive, but x2y2 should be exclusive. Thus, we increase by one.
        # There is a similar behavior in other reference implementations:
        # https://github.com/pytorch/vision/blob/b6feccbc4387766b76a3e22b13815dbbbfa87c0f/torchvision/models/detection/roi_heads.py#L418-L422
        dst_boxes_xyxy[:, 2:] += 1
        dst_boxes = F.convert_format_bounding_box(
            dst_boxes_xyxy, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=src_boxes.format, inplace=True
        )
        dst_boxes = datapoints.BoundingBox(dst_boxes, format=src_boxes.format, spatial_size=dst_spatial_size)
        boxes = datapoints.BoundingBox.wrap_like(dst_boxes, torch.cat([dst_boxes, src_boxes]))

        labels = datapoints.Label.wrap_like(dst_labels, torch.cat([dst_labels[valid_dst_masks], src_labels]))

        # Check for degenerated boxes and remove them
        # FIXME: This can only happen for the `src_boxes`, right? Since `dst_boxes` were re-computed from `dst_masks`
        #  above, they should all be valid. If so, degenerate boxes at this stage should only come from the resizing of
        #  `src_boxes` above. Maybe we can remove already at that stage?
        # TODO: Maybe unify this with `transforms.RemoveSmallBoundingBoxes()`?
        boxes_xyxy = F.convert_format_bounding_box(
            boxes, old_format=boxes.format, new_format=datapoints.BoundingBoxFormat.XYXY
        )
        degenerate_boxes = boxes_xyxy[:, 2:].le(boxes_xyxy[:, :2])
        if degenerate_boxes.any():
            valid_boxes = ~degenerate_boxes.any(dim=-1)

            masks = self._wrapping_getitem(masks, valid_boxes)
            boxes = self._wrapping_getitem(boxes, valid_boxes)
            labels = self._wrapping_getitem(labels, valid_boxes)

        return dict(image=image, masks=masks, boxes=boxes, labels=labels)


class MixupDetection(_DetectionBatchTransform):
    def __init__(
        self,
        *,
        alpha: float = 1.5,
    ) -> None:
        super().__init__()
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if has_any(flat_inputs, datapoints.Mask, datapoints.Video):
            raise TypeError(f"{type(self).__name__}() is only supported for images and bounding boxes.")

    def forward(self, *inputs: Any) -> Any:
        flat_batch_with_spec, batch = self._flatten_and_extract_data(
            inputs,
            image=(datapoints.Image, PIL.Image.Image, is_simple_tensor),
            boxes=(datapoints.BoundingBox,),
            labels=(datapoints.Label, datapoints.OneHotLabel),
        )
        self._check_inputs(flat_batch_with_spec[0])

        batch = self._to_image_tensor(batch)

        batch_output = [
            self._mixup(sample, sample_rolled, ratio=float(self._dist.sample()))
            for sample, sample_rolled in zip(batch, batch[-1:] + batch[:-1])
        ]

        return self._unflatten_and_insert_data(flat_batch_with_spec, batch_output)

    def _mixup(self, sample_1: Dict[str, Any], sample_2: Dict[str, Any], *, ratio: float) -> Dict[str, Any]:
        if ratio >= 1.0:
            return sample_1
        elif ratio == 0.0:
            return sample_2

        h_1, w_1 = sample_1["image"].shape[-2:]
        h_2, w_2 = sample_2["image"].shape[-2:]
        h_mixup = max(h_1, h_2)
        w_mixup = max(w_1, w_2)

        # TODO: add the option to fill this with something else than 0
        dtype = sample_1["image"].dtype if sample_1["image"].is_floating_point() else torch.float32
        mix_image = F.pad_image_tensor(
            sample_1["image"].to(dtype), padding=[0, 0, w_mixup - w_1, h_mixup - h_1], fill=None
        ).mul_(ratio)
        mix_image[..., :h_2, :w_2] += sample_2["image"] * (1.0 - ratio)
        mix_image = mix_image.to(sample_1["image"])

        mix_boxes = datapoints.BoundingBox.wrap_like(
            sample_1["boxes"],
            torch.cat([sample_1["boxes"], sample_2["boxes"]], dim=-2),
            spatial_size=(h_mixup, w_mixup),
        )

        mix_labels = datapoints.Label.wrap_like(
            sample_1["labels"],
            torch.cat([sample_1["labels"], sample_2["labels"]], dim=-1),
        )

        return dict(image=mix_image, boxes=mix_boxes, labels=mix_labels)
