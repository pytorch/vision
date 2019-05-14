# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from .image_list import to_image_list
from torchvision.ops import misc as misc_nn_ops


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, backbone, rpn, roi_heads):
        super(GeneralizedRCNN, self).__init__()

        self.preprocess = Transform()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None, original_image_sizes=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # images = to_image_list(images)
        images, original_image_sizes, targets = self.preprocess(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        # if original_image_sizes is not None:
        if not self.training:
            detections, _ = self.roi_heads.predict(features, proposals, images.image_sizes, original_image_sizes)
            detector_losses = {}
        else:
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections


from .image_list import ImageList
class Transform(nn.Module):
    def __init__(self, min_size=800, max_size=1333):
        super(Transform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, images, targets):
        original_image_sizes = [img.shape[-2:] for img in images]
        for i in range(len(images)):
            image = images[i]
            h, w = image.shape[-2:]
            min_size = min(image.shape[-2:])
            max_size = max(image.shape[-2:])
            scale_factor = self.min_size / min_size
            if max_size * scale_factor > self.max_size:
                scale_factor = self.max_size / max_size
            image = torch.nn.functional.interpolate(
                image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

            images[i] = image
            if targets is None:
                continue

            target = targets[i]

            bbox = target["boxes"]
            bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
            target["boxes"] = bbox

            if "masks" in target:
                mask = target["masks"]
                mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
                target["masks"] = mask

            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
                target["keypoints"] = keypoints

            targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        # return images, image_sizes, original_image_sizes, targets
        return image_list, original_image_sizes, targets

    def batch_images(self, images, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        # image_sizes = [im.shape[-2:] for im in tensors]
        return batched_imgs


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    # ratio_width, ratio_height = ratios
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
