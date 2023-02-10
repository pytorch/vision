from collections import defaultdict

import torch
import transforms as reference_transforms
from torchvision.prototype import datapoints, transforms as T


# TODO: Should we provide a transforms that filters-out keys?

# TODO: Make this public, implement it properly
class Sanitize(torch.nn.Module):
    def forward(self, inpt):
        # TODO: why are some img Tensor while others are datapoints.Image??????
        img, target = inpt
        boxes = target["boxes"]
        labels = target["labels"]

        ok_idx = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        target["boxes"] = boxes[ok_idx]  # TODO: does this preserve the DataPoint subclass?
        target["labels"] = labels[ok_idx]

        return img, target


class DetectionPresetTrain(T.Compose):
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        if data_augmentation == "hflip":
            transforms = [
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "lsj":
            transforms = [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                reference_transforms.FixedSizeCrop(
                    size=(1024, 1024), fill=defaultdict(lambda: mean, {datapoints.Mask: 0})
                ),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "multiscale":
            transforms = [
                T.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333, antialias=True
                ),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "ssd":
            transforms = [
                T.ToImageTensor(),  # Here?
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=defaultdict(lambda: mean, {datapoints.Mask: 0})),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ConvertImageDtype(torch.float),
                Sanitize(),
            ]
        elif data_augmentation == "ssdlite":
            transforms = [
                T.ToImageTensor(),  # Here?
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ConvertImageDtype(torch.float),
                Sanitize(),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        # TODO: we should convert to tensor *somewhere* (where?) because
        # otherwise `.to()` would fail during training on PIL.Image.
        # Can't do it here because ConvertImageDtype is pass-through for PIL
        # (see todo in its functional part)
        # transforms += [T.ToImageTensor()]

        super().__init__(transforms)


class DetectionPresetEval(T.Compose):
    def __init__(self):
        super().__init__(
            [
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
