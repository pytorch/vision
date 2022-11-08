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
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform
from torchvision.prototype.transforms._utils import has_any

from ._augment import flatten_and_extract_data, unflatten_and_insert_data


class MixupDetection(Transform):
    _transformed_types = (features.is_simple_tensor, features.Image, PIL.Image)

    def __init__(
        self,
        *,
        alpha: float = 1.5,
    ) -> None:
        super().__init__()
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

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
            raise TypeError(f"{type(self).__name__}() is only defined for tensor images and bounding boxes.")

    def forward(self, *inputs: Any) -> Any:
        flat_batch_with_spec, images, targets = flatten_and_extract_data(
            inputs,
            boxes=(features.BoundingBox,),
            labels=(features.Label, features.OneHotLabel),
        )
        # TODO: refactor this since we have already extracted the images and boxes
        self._check_inputs(flat_batch_with_spec[0])

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

        return unflatten_and_insert_data(flat_batch_with_spec, output_images, output_targets)

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
        mixup_ratio = self._dist.sample().item()
        print(mixup_ratio)

        c_1, h_1, w_1 = image_1.shape
        c_2, h_2, w_2 = image_2.shape
        h_mixup = max(h_1, h_2)
        w_mixup = max(w_1, w_2)

        if mixup_ratio >= 1.0:
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
