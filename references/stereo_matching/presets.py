import torch
import transforms as T


class StereoMatchingPresetEval(torch.nn.Module):
    def __init__(self, size=None) -> None:
        super().__init__()

        transforms = [
                T.ToTensor(),
                T.MakeValidDisparityMask(512),  # we keep this transform for API consistency
                T.ConvertImageDtype(torch.float32),
                T.Resize(size),
                T.Normalize(mean=0.5, std=0.5),
                T.ValidateModelInput(),
            ]
        
        if size is not None:
            transforms = transforms + [T.Resize(size)]

        self.transforms = T.Compose(transforms)

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
        # masking
        max_disparity=256,
        # AssymetricColorJitter
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=0.0,
        hue=0.0,
        #
        asymmetric_jitter_prob=1.0,
        do_flip=True,
        use_gpu=True,
    ) -> None:
        super().__init__()
        transforms = [
            T.ToTensor(),
            T.AsymmetricColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=asymmetric_jitter_prob
            ),
            T.AsymetricGammaAdjust(
                p=asymmetric_jitter_prob,
                gamma_range=(0.8, 1.2)
            ),
            T.RandomSpatialShift(),
            T.ConvertImageDtype(torch.float32),
            T.RandomResizeAndCrop(
                crop_size=crop_size,
                min_scale=min_scale,
                max_scale=max_scale,
                resize_prob=1.0
            ),
        ]

        if do_flip:
            transforms += [T.RandomHorizontalFlip()]

        transforms += [
            # occlusion after flip, otherwise we're occluding the reference image
            T.RandomOcclusion(),   
            T.Normalize(mean=0.5, std=0.5),
            T.MakeValidDisparityMask(max_disparity),
            T.ValidateModelInput(),
        ]

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparties, mask):
        return self.transforms(images, disparties, mask)

LowScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=-0.2, max_scale=0.5)
HighScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=0.6, max_scale=1.0)
SuperHighScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=0.0, max_scale=1.0)
MidScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=-0.4, max_scale=0.8)
SuperWideScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=-1.0, max_scale=1.0)
SuperLowScaleStereoMatchingPresetCRETrain = StereoMatchingPresetCRETrain(crop_size=(384, 512), min_scale=-1.0, max_scale=0.0)
MiddleBurryEvalPreset = StereoMatchingPresetEval(size=(768, 1024))

