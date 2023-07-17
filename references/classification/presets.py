import torch
from torchvision.transforms.functional import InterpolationMode


def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        module = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(module.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(module.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(module.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(module.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(module.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(module.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = module.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(module.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(module.PILToTensor())

        transforms.extend(
            [
                module.ConvertImageDtype(torch.float),
                module.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(module.RandomErasing(p=random_erase_prob))

        self.transforms = module.Compose(transforms)

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
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        module = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(module.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            module.Resize(resize_size, interpolation=interpolation, antialias=True),
            module.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(module.PILToTensor())

        transforms += [
            module.ConvertImageDtype(torch.float),
            module.Normalize(mean=mean, std=std),
        ]

        self.transforms = module.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
