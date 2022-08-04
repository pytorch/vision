import torch
import transforms as T


class StereoMatchingPresetEval(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.MakeValidDisparityMask(256),  # we keep this transform for API consistency
                T.ConvertImageDtype(torch.float32),
                T.RandomResizeAndCrop((384, 512), stretch_prob=0.0, resize_prob=0.0),
                T.Normalize(mean=0.5, std=0.5),
                T.ValidateModelInput(),
            ]
        )

    def forward(self, images, disparities, masks):
        return self.transforms(images, disparities, masks)


class StereoMatchingPresetCRETrain(torch.nn.Module):
    def __init__(
        self,
        *,
        # RandomResizeAndCrop params
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        stretch_prob=0.8,
        # masking
        max_disparity=256,
        # AssymetricColorJitter
        brightness=0.4,
        contrast=0.4,
        saturation=0.0,
        hue=0.0,
        #
        asymmetric_jitter_prob=0.5,
        do_flip=True,
    ) -> None:
        super().__init__()

        transforms = [
            T.ToTensor(),
            T.AsymmetricColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=asymmetric_jitter_prob
            ),
            T.AsymetricGammaAdjust(p=asymmetric_jitter_prob, gamma_range=(0.8, 1.2)),
            T.RandomResizeAndCrop(
                crop_size=crop_size, min_scale=min_scale, max_scale=max_scale, stretch_prob=stretch_prob
            ),
            T.ConvertImageDtype(torch.float32),
            T.RandomOcclusion(),
        ]

        if do_flip:
            transforms += [T.RandomHorizontalFlip()]

        transforms += [
            T.Normalize(mean=0.5, std=0.5),
            T.MakeValidDisparityMask(max_disparity),
            T.ValidateModelInput(),
        ]

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparties, mask):
        return self.transforms(images, disparties, mask)
