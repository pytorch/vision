import random
import torch

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        min_size = min(image.shape[-2:])
        max_size = max(image.shape[-2:])
        scale_factor = self.min_size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        mask = target["masks"]
        if mask.numel() > 0:
            mask = torch.nn.functional.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
        else:
            mask = torch.empty((0, image.shape[-2], image.shape[-1]), dtype=torch.uint8)
        bbox = target["boxes"] * scale_factor
        target["masks"] = mask
        target["boxes"] = bbox
        return image, target


# TODO finish
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class BGR255(object):
    def __call__(self, image, target):
        image = image[[2, 1, 0]] * 255
        return image, target
