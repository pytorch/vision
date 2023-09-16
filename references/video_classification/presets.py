import torch
from torchvision.transforms import transforms
from transforms import ConvertBCHWtoCBHW


class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        trans = [
            transforms.ConvertImageDtype(torch.float32),
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the video models with antialias=True?
            transforms.Resize(resize_size, antialias=False),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([transforms.Normalize(mean=mean, std=std), transforms.RandomCrop(crop_size), ConvertBCHWtoCBHW()])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, *, crop_size, resize_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                # We hard-code antialias=False to preserve results after we changed
                # its default from None to True (see
                # https://github.com/pytorch/vision/pull/7160)
                # TODO: we could re-train the video models with antialias=True?
                transforms.Resize(resize_size, antialias=False),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
