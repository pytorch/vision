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
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0), backend="PIL"):
        transforms = []

        backend = backend.lower()
        if backend == "datapoint":
            transforms.append(T.ToImageTensor())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        else:
            assert backend == "pil"

        if data_augmentation == "hflip":
            transforms += [
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                reference_transforms.FixedSizeCrop(
                    size=(1024, 1024), fill=defaultdict(lambda: mean, {datapoints.Mask: 0})
                ),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333, antialias=True
                ),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=defaultdict(lambda: mean, {datapoints.Mask: 0})),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could also just use pure tensors?
            transforms.append(T.ToImageTensor())

        transforms += [
            T.ConvertImageDtype(torch.float),
            Sanitize(),
        ]

        super().__init__(transforms)


class DetectionPresetEval(T.Compose):
    def __init__(self, backend="pil"):

        transforms = []

        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        else: #  for datapoint **and** PIL
            transforms.append(T.ToImageTensor())


        transforms.append(T.ConvertImageDtype(torch.float))
        super().__init__(transforms)