from typing import Any, cast, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes
from torchvision.prototype import tv_tensors as proto_tv_tensors
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
from torchvision.transforms.v2._utils import is_pure_tensor

from torchvision.transforms.v2.functional._geometry import _check_interpolation


class SimpleCopyPaste(Transform):
    def __init__(
        self,
        blending: bool = True,
        resize_interpolation: Union[int, InterpolationMode] = F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.resize_interpolation = _check_interpolation(resize_interpolation)
        self.blending = blending
        self.antialias = antialias

    def _copy_paste(
        self,
        image: Union[torch.Tensor, tv_tensors.Image],
        target: Dict[str, Any],
        paste_image: Union[torch.Tensor, tv_tensors.Image],
        paste_target: Dict[str, Any],
        random_selection: torch.Tensor,
        blending: bool,
        resize_interpolation: F.InterpolationMode,
        antialias: Optional[bool],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        paste_masks = tv_tensors.wrap(paste_target["masks"][random_selection], like=paste_target["masks"])
        paste_boxes = tv_tensors.wrap(paste_target["boxes"][random_selection], like=paste_target["boxes"])
        paste_labels = tv_tensors.wrap(paste_target["labels"][random_selection], like=paste_target["labels"])

        masks = target["masks"]

        # We resize source and paste data if they have different sizes
        # This is something different to TF implementation we introduced here as
        # originally the algorithm works on equal-sized data
        # (for example, coming from LSJ data augmentations)
        size1 = cast(List[int], image.shape[-2:])
        size2 = paste_image.shape[-2:]
        if size1 != size2:
            paste_image = F.resize(paste_image, size=size1, interpolation=resize_interpolation, antialias=antialias)
            paste_masks = F.resize(paste_masks, size=size1)
            paste_boxes = F.resize(paste_boxes, size=size1)

        paste_alpha_mask = paste_masks.sum(dim=0) > 0

        if blending:
            paste_alpha_mask = F.gaussian_blur(paste_alpha_mask.unsqueeze(0), kernel_size=[5, 5], sigma=[2.0])

        inverse_paste_alpha_mask = paste_alpha_mask.logical_not()
        # Copy-paste images:
        image = image.mul(inverse_paste_alpha_mask).add_(paste_image.mul(paste_alpha_mask))

        # Copy-paste masks:
        masks = masks * inverse_paste_alpha_mask
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
            xyxy_boxes, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=bbox_format, inplace=True
        )
        out_target["boxes"] = torch.cat([boxes, paste_boxes])

        labels = target["labels"][non_all_zero_masks]
        out_target["labels"] = torch.cat([labels, paste_labels])

        # Check for degenerated boxes and remove them
        boxes = F.convert_bounding_box_format(
            out_target["boxes"], old_format=bbox_format, new_format=tv_tensors.BoundingBoxFormat.XYXY
        )
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            valid_targets = ~degenerate_boxes.any(dim=1)

            out_target["boxes"] = boxes[valid_targets]
            out_target["masks"] = out_target["masks"][valid_targets]
            out_target["labels"] = out_target["labels"][valid_targets]

        return image, out_target

    def _extract_image_targets(
        self, flat_sample: List[Any]
    ) -> Tuple[List[Union[torch.Tensor, tv_tensors.Image]], List[Dict[str, Any]]]:
        # fetch all images, bboxes, masks and labels from unstructured input
        # with List[image], List[BoundingBoxes], List[Mask], List[Label]
        images, bboxes, masks, labels = [], [], [], []
        for obj in flat_sample:
            if isinstance(obj, tv_tensors.Image) or is_pure_tensor(obj):
                images.append(obj)
            elif isinstance(obj, PIL.Image.Image):
                images.append(F.to_image(obj))
            elif isinstance(obj, tv_tensors.BoundingBoxes):
                bboxes.append(obj)
            elif isinstance(obj, tv_tensors.Mask):
                masks.append(obj)
            elif isinstance(obj, (proto_tv_tensors.Label, proto_tv_tensors.OneHotLabel)):
                labels.append(obj)

        if not (len(images) == len(bboxes) == len(masks) == len(labels)):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain equal sized list of Images, "
                "BoundingBoxes, Masks and Labels or OneHotLabels."
            )

        targets = []
        for bbox, mask, label in zip(bboxes, masks, labels):
            targets.append({"boxes": bbox, "masks": mask, "labels": label})

        return images, targets

    def _insert_outputs(
        self,
        flat_sample: List[Any],
        output_images: List[torch.Tensor],
        output_targets: List[Dict[str, Any]],
    ) -> None:
        c0, c1, c2, c3 = 0, 0, 0, 0
        for i, obj in enumerate(flat_sample):
            if isinstance(obj, tv_tensors.Image):
                flat_sample[i] = tv_tensors.wrap(output_images[c0], like=obj)
                c0 += 1
            elif isinstance(obj, PIL.Image.Image):
                flat_sample[i] = F.to_pil_image(output_images[c0])
                c0 += 1
            elif is_pure_tensor(obj):
                flat_sample[i] = output_images[c0]
                c0 += 1
            elif isinstance(obj, tv_tensors.BoundingBoxes):
                flat_sample[i] = tv_tensors.wrap(output_targets[c1]["boxes"], like=obj)
                c1 += 1
            elif isinstance(obj, tv_tensors.Mask):
                flat_sample[i] = tv_tensors.wrap(output_targets[c2]["masks"], like=obj)
                c2 += 1
            elif isinstance(obj, (proto_tv_tensors.Label, proto_tv_tensors.OneHotLabel)):
                flat_sample[i] = tv_tensors.wrap(output_targets[c3]["labels"], like=obj)
                c3 += 1

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        images, targets = self._extract_image_targets(flat_inputs)

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
                    antialias=self.antialias,
                )
            output_images.append(output_image)
            output_targets.append(output_target)

        # Insert updated images and targets into input flat_sample
        self._insert_outputs(flat_inputs, output_images, output_targets)

        return tree_unflatten(flat_inputs, spec)
