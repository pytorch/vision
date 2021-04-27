import transforms as T


class DetectionPresetTrain:
    def __init__(self, hflip_prob=0.5, ssd_augmentation=False, mean=(123., 117., 104.), scaling=True):
        if ssd_augmentation:
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list(mean)),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(scaling=scaling),
            ])
        else:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(scaling=scaling),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, scaling=True):
        self.transforms = T.ToTensor(scaling=scaling)

    def __call__(self, img, target):
        return self.transforms(img, target)
