import torch
from torchvision.prototype import transforms


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=transforms.InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
    ):
        trans = []

        backend = backend.lower()
        if backend == "datapoint":
            trans.append(transforms.ToImageTensor())
        elif backend == "tensor":
            trans.append(transforms.PILToTensor())
        else:
            assert backend == "pil"

        trans.append(transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
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

        if backend == "pil":
            # Note: we could also just use pure tensors?
            trans.append(transforms.ToImageTensor())

        trans.extend(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=transforms.InterpolationMode.BILINEAR,
        backend="pil",
    ):
        trans = []

        backend = backend.lower()
        if backend == "datapoint":
            trans.append(transforms.ToImageTensor())
        elif backend == "tensor":
            trans.append(transforms.PILToTensor())
        else:
            assert backend == "pil"

        trans += [
            transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
        ]

        if backend == "pil":
            trans.append(transforms.ToImageTensor())

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
