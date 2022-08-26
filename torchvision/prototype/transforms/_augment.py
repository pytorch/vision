import math
import numbers
import warnings
from typing import Any, Dict, List, Tuple

import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.ops import masks_to_boxes
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, InterpolationMode

from ._transform import _RandomApplyTransform
from ._utils import has_any, query_chw


class RandomErasing(_RandomApplyTransform):
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
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
        self.value = value

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        img_c, img_h, img_w = query_chw(sample)

        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img_c)):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of inpt channels)"
            )

        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(self.ratio))
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

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break
        else:
            i, j, h, w, v = 0, 0, img_h, img_w, None

        return dict(i=i, j=j, h=h, w=w, v=v)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["v"] is not None:
            inpt = F.erase(inpt, **params)

        return inpt


class _BaseMixupCutmix(_RandomApplyTransform):
    def __init__(self, *, alpha: float, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def forward(self, *inputs: Any) -> Any:
        if not (has_any(inputs, features.Image, features.is_simple_tensor) and has_any(inputs, features.OneHotLabel)):
            raise TypeError(f"{type(self).__name__}() is only defined for tensor images and one-hot labels.")
        if has_any(inputs, features.BoundingBox, features.SegmentationMask, features.Label):
            raise TypeError(
                f"{type(self).__name__}() does not support bounding boxes, segmentation masks and plain labels."
            )
        return super().forward(*inputs)

    def _mixup_onehotlabel(self, inpt: features.OneHotLabel, lam: float) -> features.OneHotLabel:
        if inpt.ndim < 2:
            raise ValueError("Need a batch of one hot labels")
        output = inpt.clone()
        output = output.roll(1, -2).mul_(1 - lam).add_(output.mul_(lam))
        return features.OneHotLabel.new_like(inpt, output)


class RandomMixup(_BaseMixupCutmix):
    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]
        if isinstance(inpt, features.Image) or features.is_simple_tensor(inpt):
            if inpt.ndim < 4:
                raise ValueError("Need a batch of images")
            output = inpt.clone()
            output = output.roll(1, -4).mul_(1 - lam).add_(output.mul_(lam))

            if isinstance(inpt, features.Image):
                output = features.Image.new_like(inpt, output)

            return output
        elif isinstance(inpt, features.OneHotLabel):
            return self._mixup_onehotlabel(inpt, lam)
        else:
            return inpt


class RandomCutmix(_BaseMixupCutmix):
    def _get_params(self, sample: Any) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))

        _, H, W = query_chw(sample)

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
        if isinstance(inpt, features.Image) or features.is_simple_tensor(inpt):
            box = params["box"]
            if inpt.ndim < 4:
                raise ValueError("Need a batch of images")
            x1, y1, x2, y2 = box
            image_rolled = inpt.roll(1, -4)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = image_rolled[..., y1:y2, x1:x2]

            if isinstance(inpt, features.Image):
                output = features.Image.new_like(inpt, output)

            return output
        elif isinstance(inpt, features.OneHotLabel):
            lam_adjusted = params["lam_adjusted"]
            return self._mixup_onehotlabel(inpt, lam_adjusted)
        else:
            return inpt


class SimpleCopyPaste(_RandomApplyTransform):
    def __init__(
        self,
        p: float = 0.5,
        blending: bool = True,
        resize_interpolation: InterpolationMode = F.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__(p=p)
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def _copy_paste(
        self,
        image: Any,
        target: Dict[str, Any],
        paste_image: Any,
        paste_target: Dict[str, Any],
        random_selection: torch.Tensor,
        blending: bool = True,
        resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
    ) -> Tuple[Any, Dict[str, Any]]:

        paste_masks = paste_target["masks"].new_like(paste_target["masks"], paste_target["masks"][random_selection])
        paste_boxes = paste_target["boxes"].new_like(paste_target["boxes"], paste_target["boxes"][random_selection])
        paste_labels = paste_target["labels"].new_like(paste_target["labels"], paste_target["labels"][random_selection])

        masks = target["masks"]

        # We resize source and paste data if they have different sizes
        # This is something different to TF implementation we introduced here as
        # originally the algorithm works on equal-sized data
        # (for example, coming from LSJ data augmentations)
        size1 = image.shape[-2:]
        size2 = paste_image.shape[-2:]
        if size1 != size2:
            paste_image = F.resize(paste_image, size=size1, interpolation=resize_interpolation)
            paste_masks = F.resize(paste_masks, size=size1)
            paste_boxes = F.resize(paste_boxes, size=size1)

        paste_alpha_mask = paste_masks.sum(dim=0) > 0

        if blending:
            paste_alpha_mask = F.gaussian_blur(paste_alpha_mask.unsqueeze(0), kernel_size=[5, 5], sigma=[2.0])

        # Copy-paste images:
        image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

        # Copy-paste masks:
        masks = masks * (~paste_alpha_mask)
        non_all_zero_masks = masks.sum((-1, -2)) > 0
        masks = masks[non_all_zero_masks]

        # Do a shallow copy of the target dict
        out_target = {k: v for k, v in target.items()}

        out_target["masks"] = torch.cat([masks, paste_masks])

        # Copy-paste boxes and labels
        bbox_format = target["boxes"].format
        xyxy_boxes = masks_to_boxes(masks)
        # masks_to_boxes produces bboxes with x2y2 inclusive but x2y2 should be exclusive
        # we need to add +1 to x2y2.
        # There is a similar +1 in other reference implementations:
        # https://github.com/pytorch/vision/blob/b6feccbc4387766b76a3e22b13815dbbbfa87c0f/torchvision/models/detection/roi_heads.py#L418-L422
        xyxy_boxes[:, 2:] += 1
        boxes = F.convert_bounding_box_format(
            xyxy_boxes, old_format=features.BoundingBoxFormat.XYXY, new_format=bbox_format, copy=False
        )
        out_target["boxes"] = torch.cat([boxes, paste_boxes])

        labels = target["labels"][non_all_zero_masks]
        out_target["labels"] = torch.cat([labels, paste_labels])

        # Check for degenerated boxes and remove them
        boxes = F.convert_bounding_box_format(
            out_target["boxes"], old_format=bbox_format, new_format=features.BoundingBoxFormat.XYXY
        )
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            valid_targets = ~degenerate_boxes.any(dim=1)

            out_target["boxes"] = boxes[valid_targets]
            out_target["masks"] = out_target["masks"][valid_targets]
            out_target["labels"] = out_target["labels"][valid_targets]

        return image, out_target

    def _extract_image_targets(self, flat_sample: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        # fetch all images, bboxes, masks and labels from unstructured input
        # with List[image], List[BoundingBox], List[SegmentationMask], List[Label]
        images, bboxes, masks, labels = [], [], [], []
        for obj in flat_sample:
            if isinstance(obj, features.Image) or features.is_simple_tensor(obj):
                images.append(obj)
            elif isinstance(obj, PIL.Image.Image):
                images.append(F.to_image_tensor(obj))
            elif isinstance(obj, features.BoundingBox):
                bboxes.append(obj)
            elif isinstance(obj, features.SegmentationMask):
                masks.append(obj)
            elif isinstance(obj, (features.Label, features.OneHotLabel)):
                labels.append(obj)

        if not (len(images) == len(bboxes) == len(masks) == len(labels)):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain equal sized list of Images, "
                "BoundingBoxes, Segmentation Masks and Labels or OneHotLabels."
            )

        targets = []
        for bbox, mask, label in zip(bboxes, masks, labels):
            targets.append({"boxes": bbox, "masks": mask, "labels": label})

        return images, targets

    def _insert_outputs(
        self, flat_sample: List[Any], output_images: List[Any], output_targets: List[Dict[str, Any]]
    ) -> None:
        c0, c1, c2, c3 = 0, 0, 0, 0
        for i, obj in enumerate(flat_sample):
            if isinstance(obj, features.Image):
                flat_sample[i] = features.Image.new_like(obj, output_images[c0])
                c0 += 1
            elif isinstance(obj, PIL.Image.Image):
                flat_sample[i] = F.to_image_pil(output_images[c0])
                c0 += 1
            elif features.is_simple_tensor(obj):
                flat_sample[i] = output_images[c0]
                c0 += 1
            elif isinstance(obj, features.BoundingBox):
                flat_sample[i] = features.BoundingBox.new_like(obj, output_targets[c1]["boxes"])
                c1 += 1
            elif isinstance(obj, features.SegmentationMask):
                flat_sample[i] = features.SegmentationMask.new_like(obj, output_targets[c2]["masks"])
                c2 += 1
            elif isinstance(obj, (features.Label, features.OneHotLabel)):
                flat_sample[i] = obj.new_like(obj, output_targets[c3]["labels"])  # type: ignore[arg-type]
                c3 += 1

    def forward(self, *inputs: Any) -> Any:
        flat_sample, spec = tree_flatten(inputs)

        images, targets = self._extract_image_targets(flat_sample)

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [t2, t3, ..., tN, t1]
        # FYI: in TF they mix data on the dataset level
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images, output_targets = [], []

        for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):

            # Random paste targets selection:
            num_masks = len(paste_target["masks"])

            if num_masks < 1:
                # Such degerante case with num_masks=0 can happen with LSJ
                # Let's just return (image, target)
                output_image, output_target = image, target
            else:
                random_selection = torch.randint(0, num_masks, (num_masks,), device=paste_image.device)
                random_selection = torch.unique(random_selection)

                output_image, output_target = self._copy_paste(
                    image,
                    target,
                    paste_image,
                    paste_target,
                    random_selection=random_selection,
                    blending=self.blending,
                    resize_interpolation=self.resize_interpolation,
                )
            output_images.append(output_image)
            output_targets.append(output_target)

        # Insert updated images and targets into input flat_sample
        self._insert_outputs(flat_sample, output_images, output_targets)

        return tree_unflatten(flat_sample, spec)
