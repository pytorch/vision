from typing import Optional, Tuple, Union

import torch
import transforms as T


class StereoMatchingEvalPreset(torch.nn.Module):
    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.5,
        resize_size: Optional[Tuple[int, int]] = None,
        max_disparity: Optional[float] = 512,
        interpolation_type: str = "bilinear",
    ) -> None:
        super().__init__()

        transforms = [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=mean, std=std),
            T.MakeValidDisparityMask(max_disparity),  # we keep this transform for API consistency
            T.ValidateModelInput(),
        ]

        if resize_size is not None:
            transforms.append(T.Resize(resize_size, interpolation_type=interpolation_type))

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparities, masks):
        return self.transforms(images, disparities, masks)


class StereoMatchingTrainPreset(torch.nn.Module):
    def __init__(
        self,
        *,
        # RandomResizeAndCrop params
        crop_size: Tuple[int, int],
        resize_prob: float = 1.0,
        scaling_type: str = "exponential",
        scale_range: Tuple[float, float] = (-0.2, 0.5),
        scale_interpolation_type: str = "bilinear",
        # normalization params:
        mean: float = 0.5,
        std: float = 0.5,
        # processing device
        gpu_transforms: bool = False,
        # masking
        max_disparity: Optional[int] = 256,
        # SpatialShift params
        spatial_shift_prob: float = 0.5,
        spatial_shift_max_angle: float = 0.5,
        spatial_shift_max_displacement: float = 0.5,
        spatial_shift_interpolation_type: str = "bilinear",
        # AssymetricColorJitter
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        brightness: Union[int, Tuple[int, int]] = (0.8, 1.2),
        contrast: Union[int, Tuple[int, int]] = (0.8, 1.2),
        saturation: Union[int, Tuple[int, int]] = 0.0,
        hue: Union[int, Tuple[int, int]] = 0.0,
        asymmetric_jitter_prob: float = 1.0,
        # RandomHorizontalFlip
        horizontal_flip_prob: float = 0.5,
        # RandomOcclusion
        occlusion_prob: float = 0.0,
        occlusion_px_range: Tuple[int, int] = (50, 100),
        # RandomErase
        erase_prob: float = 0.0,
        erase_px_range: Tuple[int, int] = (50, 100),
        erase_num_repeats: int = 1,
    ) -> None:

        if scaling_type not in ["linear", "exponential"]:
            raise ValueError(f"Unknown scaling type: {scaling_type}. Available types: linear, exponential")

        super().__init__()
        transforms = [T.ToTensor()]
        if gpu_transforms:
            transforms.append(T.ToGPU())

        transforms.extend(
            [
                T.AsymmetricColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=asymmetric_jitter_prob
                ),
                T.AsymetricGammaAdjust(p=asymmetric_jitter_prob, gamma_range=gamma_range),
                T.RandomSpatialShift(
                    p=spatial_shift_prob,
                    max_angle=spatial_shift_max_angle,
                    max_displacement=spatial_shift_max_displacement,
                    interpolation_type=spatial_shift_interpolation_type,
                ),
                T.ConvertImageDtype(torch.float32),
                T.RandomResizeAndCrop(
                    crop_size=crop_size,
                    scale_range=scale_range,
                    resize_prob=resize_prob,
                    scaling_type=scaling_type,
                    interpolation_type=scale_interpolation_type,
                ),
                T.RandomHorizontalFlip(horizontal_flip_prob),
                # occlusion after flip, otherwise we're occluding the reference image
                T.RandomOcclusion(p=occlusion_prob, occlusion_px_range=occlusion_px_range),
                T.RandomErase(p=erase_prob, erase_px_range=erase_px_range, num_repeats=erase_num_repeats),
                T.Normalize(mean=mean, std=std),
                T.MakeValidDisparityMask(max_disparity),
                T.ValidateModelInput(),
            ]
        )

        self.transforms = T.Compose(transforms)

    def forward(self, images, disparties, mask):
        return self.transforms(images, disparties, mask)
