"""
This script is used to apply the mixup transform for Object detection to the dataset.
The code is inspired from the paper: https://arxiv.org/abs/1902.0410

In a nutshell, mixup is a data augmentation technique that combines two images in the ratio of
beta to (1-beta) and this value of beta is sampled from a beta distribution. This makes our model
robust to the object being present in the image or not. Plus, it is kind of like a free lunch.
"""
from typing import Any, Dict, List, Tuple

import PIL.Image

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.prototype import features
from torchvision.prototype.features._feature import is_simple_tensor
from torchvision.prototype.transforms import functional as F, Transform
from torchvision.prototype.transforms._utils import has_any


class MixupDetection(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image, PIL.Image)

    def __init__(
        self,
        *,
        alpha: float = 1.5,
    ) -> None:
        super().__init__()
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _get_params(self) -> Dict[str, Any]:
        # TODO: Retrieve the params from the input sample
        mixup_ratio = self._dist.sample().item()
        return {"mixup_ratio": mixup_ratio}

    def _extract_image_targets(self, flat_sample: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        # fetch all images, bboxes and labels from unstructured input
        # with List[image], List[BoundingBox], List[Label]
        images, bboxes, labels = [], [], []
        for obj in flat_sample:
            if isinstance(obj, features.Image) or is_simple_tensor(obj):
                images.append(obj)
            elif isinstance(obj, PIL.Image.Image):
                images.append(F.to_tensor(obj))
            elif isinstance(obj, features.BoundingBox):
                bboxes.append(obj)
            elif isinstance(obj, (features.Label, features.OneHotLabel)):
                labels.append(obj)

        if not (len(images) == len(bboxes) == len(labels)):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain equal-sized list of Images, "
                "BoundingBoxes, and Labels or OneHotLabels."
            )

        targets = []
        for bbox, label in zip(bboxes, labels):
            targets.append({"boxes": bbox, "labels": label})

        return images, targets

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if has_any(flat_inputs, features.Mask):
            raise TypeError(f"Masks are not supported by {type(self).__name__}()")

        if not has_any(flat_inputs, PIL.Image.Image, features.Image, features.is_simple_tensor):
            raise TypeError(
                f"{type(self).__name__}() requires input sample to contain an tensor or PIL image or a Video."
            )

        if not (
            has_any(flat_inputs, features.Image, PIL.Image.Image, features.is_simple_tensor)
            and has_any(flat_inputs, features.BoundingBox)
        ):
            raise TypeError(f"{type(self).__name__}() is only defined for tensor images/videos and bounding boxes.")

    def _insert_outputs(
        self, flat_sample: List[Any], output_images: List[Any], output_targets: List[Dict[str, Any]]
    ) -> None:
        c0, c1, c2 = 0, 0, 0
        for i, obj in enumerate(flat_sample):
            if isinstance(obj, features.Image):
                flat_sample[i] = features.Image.wrap_like(obj, output_images[c0])
                c0 += 1
            elif isinstance(obj, PIL.Image.Image):
                flat_sample[i] = F.to_image_pil(output_images[c0])
                c0 += 1
            elif is_simple_tensor(obj):
                flat_sample[i] = output_images[c0]
                c0 += 1
            elif isinstance(obj, features.BoundingBox):
                flat_sample[i] = features.BoundingBox.wrap_like(obj, output_targets[c1]["boxes"])
                c1 += 1
            elif isinstance(obj, (features.Label, features.OneHotLabel)):
                flat_sample[i] = obj.wrap_like(obj, output_targets[c2]["labels"])  # type: ignore[arg-type]
                c2 += 1

    def forward(self, *inputs: Any) -> Any:
        flat_sample, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        self._check_inputs(flat_sample)

        images, targets = self._extract_image_targets(flat_sample)

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [tN, t1, ..., tN-1,]
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images, output_targets = [], []
        for image_1, target_1, image_2, target_2 in zip(images, targets, images_rolled, targets_rolled):
            output_image, output_target = self._mixup(
                image_1,
                target_1,
                image_2,
                target_2,
            )
            output_images.append(output_image)
            output_targets.append(output_target)

        # Insert updated images and targets into input flat_sample
        self._insert_outputs(flat_sample, output_images, output_targets)
        return tree_unflatten(flat_sample, spec)

    def _mixup(
        self,
        image_1: features.TensorImageType,
        target_1: Dict[str, Any],
        image_2: features.TensorImageType,
        target_2: Dict[str, Any],
    ) -> Tuple[features.TensorImageType, Dict[str, Any]]:
        """
        Performs mixup on the given images and targets.
        """
        mixup_ratio = self._get_params().get("mixup_ratio")
        c_1, h_1, w_1 = image_1.shape
        c_2, h_2, w_2 = image_2.shape
        h_mixup = max(h_1, h_2)
        w_mixup = max(w_1, w_2)

        if mixup_ratio >= 1:
            return image_1, target_1

        # mixup images and prevent the object aspect ratio from changing
        mix_img = torch.zeros(c_1, h_mixup, w_mixup, dtype=torch.float32)
        mix_img[:, : image_1.shape[1], : image_1.shape[2]] = image_1 * mixup_ratio
        mix_img[:, : image_2.shape[1], : image_2.shape[2]] += image_2 * (1.0 - mixup_ratio)
        # mixup targets
        mix_target = {**target_1, **target_2}
        box_format = target_1["boxes"].format
        mixed_boxes = {
            "boxes": features.BoundingBox(
                torch.vstack((target_1["boxes"], target_2["boxes"])),
                format=box_format,
                spatial_size=(h_mixup, w_mixup),
            )
        }
        mix_labels = {"labels": torch.cat((target_1["labels"], target_2["labels"]))}
        mix_target.update(mixed_boxes)
        mix_target.update(mix_labels)

        return mix_img, mix_target
