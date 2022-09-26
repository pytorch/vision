import argparse
from functools import partial

import torch

from presets import StereoMatchingEvalPreset, StereoMatchingTrainPreset
from torchvision.datasets import (
    CarlaStereo,
    CREStereo,
    ETH3DStereo,
    FallingThingsStereo,
    InStereo2k,
    Kitti2012Stereo,
    Kitti2015Stereo,
    Middlebury2014Stereo,
    SceneFlowStereo,
    SintelStereo,
)

VALID_DATASETS = {
    "crestereo": partial(CREStereo),
    "carla-highres": partial(CarlaStereo),
    "instereo2k": partial(InStereo2k),
    "sintel": partial(SintelStereo),
    "sceneflow-monkaa": partial(SceneFlowStereo, variant="Monkaa", pass_name="both"),
    "sceneflow-flyingthings": partial(SceneFlowStereo, variant="FlyingThings3D", pass_name="both"),
    "sceneflow-driving": partial(SceneFlowStereo, variant="Driving", pass_name="both"),
    "fallingthings": partial(FallingThingsStereo, variant="both"),
    "eth3d-train": partial(ETH3DStereo, split="train"),
    "eth3d-test": partial(ETH3DStereo, split="test"),
    "kitti2015-train": partial(Kitti2015Stereo, split="train"),
    "kitti2015-test": partial(Kitti2015Stereo, split="test"),
    "kitti2012-train": partial(Kitti2012Stereo, split="train"),
    "kitti2012-test": partial(Kitti2012Stereo, split="train"),
    "middlebury2014-other": partial(
        Middlebury2014Stereo, split="additional", use_ambient_view=True, calibration="both"
    ),
    "middlebury2014-train": partial(Middlebury2014Stereo, split="train", calibration="perfect"),
    "middlebury2014-test": partial(Middlebury2014Stereo, split="test", calibration=None),
    "middlebury2014-train-ambient": partial(
        Middlebury2014Stereo, split="train", use_ambient_views=True, calibrartion="perfect"
    ),
}


def make_train_transform(args: argparse.Namespace) -> torch.nn.Module:
    return StereoMatchingTrainPreset(
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        rescale_prob=args.rescale_prob,
        scaling_type=args.scaling_type,
        scale_range=args.scale_range,
        scale_interpolation_type=args.interpolation_strategy,
        use_grayscale=args.use_grayscale,
        mean=args.norm_mean,
        std=args.norm_std,
        horizontal_flip_prob=args.flip_prob,
        gpu_transforms=args.gpu_transforms,
        max_disparity=args.max_disparity,
        spatial_shift_prob=args.spatial_shift_prob,
        spatial_shift_max_angle=args.spatial_shift_max_angle,
        spatial_shift_max_displacement=args.spatial_shift_max_displacement,
        spatial_shift_interpolation_type=args.interpolation_strategy,
        gamma_range=args.gamma_range,
        brightness=args.brightness_range,
        contrast=args.contrast_range,
        saturation=args.saturation_range,
        hue=args.hue_range,
        asymmetric_jitter_prob=args.asymmetric_jitter_prob,
    )


def make_eval_transform(args: argparse.Namespace) -> torch.nn.Module:
    if args.eval_size is None:
        resize_size = args.crop_size
    else:
        resize_size = args.eval_size

    return StereoMatchingEvalPreset(
        mean=args.norm_mean,
        std=args.norm_std,
        use_grayscale=args.use_grayscale,
        resize_size=resize_size,
        interpolation_type=args.interpolation_strategy,
    )


def make_dataset(dataset_name: str, dataset_root: str, transforms: torch.nn.Module) -> torch.utils.data.Dataset:
    return VALID_DATASETS[dataset_name](root=dataset_root, transforms=transforms)
