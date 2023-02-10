import torch
from torchvision.prototype import transforms


class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        interpolation=transforms.InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [
            transforms.RandomShortestSize(
                min_size=list(range(resize_size[0], resize_size[1] + 1)), interpolation=interpolation, antialias=True
            ),
            transforms.RandomCrop(crop_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(p=hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(transforms.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(transforms.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = transforms.AutoAugmentPolicy(auto_augment_policy)
                trans.append(transforms.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))
        trans.append(transforms.TransposeDimensions((-3, -4)))

        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
                transforms.CenterCrop(crop_size),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=mean, std=std),
                transforms.TransposeDimensions((-3, -4)),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
