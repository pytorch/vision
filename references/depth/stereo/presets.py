from typing import Tuple, Union

import torch
import transforms as T


class StereoMatchingEvalPreset(torch.nn.Module):
    def __init__(self, size=None) -> None:
        super().__init__()

        transforms = [
            T.ToTensor(),
            T.MakeValidDisparityMask(512),  # we keep this transform for API consistency
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
            T.ValidateModelInput(),
        ]

        if size is not None:
            transforms = transforms + [T.Resize(size)]

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparities, masks):
        return self.transforms(images, disparities, masks)


class StereoMatchingTrainPreset(torch.nn.Module):
    def __init__(
        self,
        *,
        # RandomResizeAndCrop params
        crop_size: Tuple[int, int],
        min_scale: float = -0.2,
        max_scale: float = 0.5,
        resize_prob: float = 1.0,
        scaling_type: str = "exponential",
        # processing device
        gpu_transforms=False,
        # masking
        max_disparity: int = 256,
        # AssymetricColorJitter
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        brightness: Union[int, Tuple[int, int]] = (0.8, 1.2),
        contrast: Union[int, Tuple[int, int]] = (0.8, 1.2),
        saturation: Union[int, Tuple[int, int]] = 0.0,
        hue: Union[int, Tuple[int, int]] = 0.0,
        asymmetric_jitter_prob: float = 1.0,
        # RandomHorizontalFlip
        do_flip=True,
        # RandomOcclusion
        occlusion_prob: float = 0.0,
        occlusion_min_px: int = 50,
        occlusion_max_px: int = 100,
        # RandomErase
        erase_prob: float = 0.0,
        erase_min_px: int = 50,
        erase_max_px: int = 100,
        erase_num_repeats: int = 1,
    ) -> None:

        if scaling_type not in ["linear", "exponential"]:
            raise ValueError(f"Unknown scaling type: {scaling_type}. Available types: linear, exponential")

        super().__init__()
        transforms = [T.ToTensor()]
        if gpu_transforms:
            transforms.append(T.ToGPU())

        transforms = [
            T.AsymmetricColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=asymmetric_jitter_prob
            ),
            T.AsymetricGammaAdjust(p=asymmetric_jitter_prob, gamma_range=gamma_range),
            T.RandomSpatialShift(),
            T.ConvertImageDtype(torch.float32),
            T.RandomResizeAndCrop(
                crop_size=crop_size,
                min_scale=min_scale,
                max_scale=max_scale,
                resize_prob=resize_prob,
                scaling_type=scaling_type,
            ),
        ]

        if do_flip:
            transforms += [T.RandomHorizontalFlip()]

        transforms += [
            # occlusion after flip, otherwise we're occluding the reference image
            T.RandomOcclusion(p=occlusion_prob, min_px=occlusion_min_px, max_px=occlusion_max_px),
            T.RandomErase(p=erase_prob, min_px=erase_min_px, max_px=erase_max_px, num_repeats=erase_num_repeats),
            T.Normalize(mean=0.5, std=0.5),
            T.MakeValidDisparityMask(max_disparity),
            T.ValidateModelInput(),
        ]

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparties, mask):
        return self.transforms(images, disparties, mask)
