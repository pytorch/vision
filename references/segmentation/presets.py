from collections import defaultdict

import torch
from torchvision.prototype import features, transforms as T
from transforms import PadIfSmaller, WrapIntoFeatures


class SegmentationPresetTrain(T.Compose):
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        transforms = [
            WrapIntoFeatures(),
            T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size), antialias=True),
        ]
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        transforms.extend(
            [
                # We need a custom pad transform here, since the padding we want to perform here is fundamentally
                # different from the padding in `RandomCrop` if `pad_if_needed=True`.
                PadIfSmaller(crop_size, fill=defaultdict(lambda: 0, {features.Mask: 255})),
                T.RandomCrop(crop_size),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        super().__init__(transforms)


class SegmentationPresetEval(T.Compose):
    def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__(
            [
                WrapIntoFeatures(),
                T.Resize(base_size, antialias=True),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
