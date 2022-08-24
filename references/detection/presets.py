import torch
from torchvision.prototype import transforms as T


class DetectionPresetTrain(T.Compose):
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        if data_augmentation == "hflip":
            transforms = [
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "lsj":
            transforms = [
                T.ScaleJitter(target_size=(1024, 1024)),
                T.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "multiscale":
            transforms = [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "ssd":
            transforms = [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list(mean)),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        elif data_augmentation == "ssdlite":
            transforms = [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        super().__init__(transforms)


class DetectionPresetEval(T.Compose):
    def __init__(self):
        super().__init__(
            [
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
