import torch
from torchvision.prototype import features, transforms as T
from torchvision.prototype.transforms import functional as F
from transforms import RandomCrop


class WrapIntoFeatures(T.Transform):
    def forward(self, sample):
        image, mask = sample
        return image, features.Mask(F.to_image_tensor(mask).squeeze(0), dtype=torch.int64)


class SegmentationPresetTrain(T.Compose):
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        transforms = [
            WrapIntoFeatures(),
            T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size)),
        ]
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        transforms.extend(
            [
                RandomCrop(crop_size, pad_if_needed=True),
                T.ToImageTensor(),
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
                T.Resize(base_size),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
