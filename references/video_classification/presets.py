import torch

from torchvision.transforms import transforms
from transforms import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW


class VideoClassificationPresetTrain:
    def __init__(self, resize_size, crop_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989),
                 hflip_prob=0.5):
        trans = [
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, resize_size, crop_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose([
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
            transforms.Normalize(mean=mean, std=std),
            transforms.CenterCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])

    def __call__(self, x):
        return self.transforms(x)
