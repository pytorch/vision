import argparse
import torch
from torchvision.datasets import (
    CREStereo,
    CarlaStereo,
    InStereo2k,
    SintelStereo,
    SceneFlowStereo,
    FallingThingsStereo,
    ETH3DStereo,
    Kitti2012Stereo,
    Kitti2015Stereo,
    Middlebury2014Stereo,
)

from presets import StereoMatchingTrainPreset, StereoMatchingEvalPreset


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
    valid_datasets = {
        "crestereo": CREStereo(root=dataset_root, transforms=transforms),
        "carla-highres": CarlaStereo(root=dataset_root, transforms=transforms),
        "instereo2k": InStereo2k(root=dataset_root, transforms=transforms),
        "sintel": SintelStereo(root=dataset_root, transforms=transforms),
        "sceneflow-monkaa": SceneFlowStereo(root=dataset_root, transforms=transforms, variant="Monkaa", pass_name="both"),
        "sceneflow-flyingthings": SceneFlowStereo(root=dataset_root, transforms=transforms, variant="FlyingThings3D", pass_name="both"),
        "sceneflow-driving": SceneFlowStereo(root=dataset_root, transforms=transforms, variant="Driving", pass_name="both"),
        "fallingthings": FallingThingsStereo(root=dataset_root, transforms=transforms, variant="both"),
        "eth3d-train": ETH3DStereo(root=dataset_root, transforms=transforms, split="train"),
        "eth3d-test": ETH3DStereo(root=dataset_root, transforms=transforms, split="test"),
        "kitti2015-train": Kitti2015Stereo(root=dataset_root, transforms=transforms, split="train"),
        "kitti2015-test": Kitti2015Stereo(root=dataset_root, transforms=transforms, split="test"),
        "kitti2012-train": Kitti2012Stereo(root=dataset_root, transforms=transforms, split="train"),
        "kitti2012-test": Kitti2012Stereo(root=dataset_root, transforms=transforms, split="test"),
        "middlebury2014-other": Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="additional", use_ambient_views=True, calibration="both"),
        "middlebury2014-train": Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="train", calibration="perfect"),
        "middlebury2014-test": Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="test", calibration=None),
        "middlebury2014-train-ambient": Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="train", use_ambient_views=True, calibration="perfect"),
    }

    # raise a key-error just to inform the user about which datasets are valid
    if dataset_name not in valid_datasets:
        raise KeyError(f"Invalid dataset name: {dataset_name}. Valid datasets are: {valid_datasets.keys()}")

    return valid_datasets[dataset_name]
        