import torch
import transforms as T


class DetectionPresetTrain:
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        if data_augmentation == "hflip":
            self.transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "lsj":
            self.transforms = T.Compose(
                [
                    T.ScaleJitter(target_size=(1024, 1024)),
                    T.FixedSizeCrop(size=(1024, 1024), fill=mean),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "multiscale":
            self.transforms = T.Compose(
                [
                    T.RandomShortestSize(
                        min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                    ),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssd":
            self.transforms = T.Compose(
                [
                    T.RandomPhotometricDistort(),
                    T.RandomZoomOut(fill=list(mean)),
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssdlite":
            self.transforms = T.Compose(
                [
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)
