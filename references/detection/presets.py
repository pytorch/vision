from collections import defaultdict

import torch
import transforms as reference_transforms


def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.datapoints
        import torchvision.transforms.v2

        return torchvision.transforms.v2, torchvision.datapoints
    else:
        return reference_transforms, None


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        use_v2=False,
    ):

        T, datapoints = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "datapoint":
            transforms.append(T.ToImageTensor())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'datapoint', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {datapoints.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImageTensor() if use_v2 else T.PILToTensor()]

        transforms += [T.ConvertImageDtype(torch.float)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(datapoints.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
            ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImageTensor() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "datapoint":
            transforms += [T.ToImageTensor()]
        else:
            raise ValueError(f"backend can be 'datapoint', 'tensor' or 'pil', but got {backend}")

        transforms += [T.ConvertImageDtype(torch.float)]
        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
