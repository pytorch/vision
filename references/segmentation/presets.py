from collections import defaultdict

import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision import datapoints
from transforms import PadIfSmaller, WrapIntoFeatures


class SegmentationPresetTrain(T.Compose):
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        backend="pil",
    ):

        transforms = []

        transforms.append(WrapIntoFeatures())

        backend = backend.lower()
        if backend == "datapoint":
            transforms.append(T.ToImageTensor())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        else:
            assert backend == "pil"

        transforms.append(T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size), antialias=True))

        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))

        transforms += [
            # We need a custom pad transform here, since the padding we want to perform here is fundamentally
            # different from the padding in `RandomCrop` if `pad_if_needed=True`.
            PadIfSmaller(crop_size, fill=defaultdict(lambda: 0, {datapoints.Mask: 255})),
            T.RandomCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.ToImageTensor())

        transforms += [
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]

        super().__init__(transforms)


class SegmentationPresetEval(T.Compose):
    def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil"):
        transforms = []

        transforms.append(WrapIntoFeatures())

        backend = backend.lower()
        if backend == "datapoint":
            transforms.append(T.ToImageTensor())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        else:
            assert backend == "pil"

        transforms.append(T.Resize(base_size, antialias=True))

        if backend == "pil":
            transforms.append(T.ToImageTensor())

        transforms += [
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]
        super().__init__(transforms)
