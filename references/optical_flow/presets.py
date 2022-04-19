import torch
import transforms as T


class OpticalFlowPresetEval(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.ValidateModelInput(),
            ]
        )

    def forward(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)


class OpticalFlowPresetTrain(torch.nn.Module):
    def __init__(
        self,
        *,
        # RandomResizeAndCrop params
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        stretch_prob=0.8,
        # AsymmetricColorJitter params
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14,
        # Random[H,V]Flip params
        asymmetric_jitter_prob=0.2,
        do_flip=True,
    ):
        super().__init__()

        transforms = [
            T.PILToTensor(),
            T.AsymmetricColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=asymmetric_jitter_prob
            ),
            T.RandomResizeAndCrop(
                crop_size=crop_size, min_scale=min_scale, max_scale=max_scale, stretch_prob=stretch_prob
            ),
        ]

        if do_flip:
            transforms += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.1)]

        transforms += [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.RandomErasing(max_erase=2),
            T.MakeValidFlowMask(),
            T.ValidateModelInput(),
        ]
        self.transforms = T.Compose(transforms)

    def forward(self, img1, img2, flow, valid):
        return self.transforms(img1, img2, flow, valid)
