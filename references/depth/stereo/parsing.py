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
    valid_datasets = [
        "crestereo",
        "carla-highres",
        "instereo2k",
        "sintel",
        "sceneflow-monkaa",
        "sceneflow-flyingthings",
        "sceneflow-driving",
        "fallingthings",
        "eth3d-train",
        "eth3d-test",
        "kitti2015-train",
        "kitti2015-test",
        "kitti2012-train",
        "kitti2012-test",
        "middlebury2014-other",
        "middlebury2014-train",
        "middlebury2014-test",
        "middlebury2014-train-ambient",
    ]

    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Valid datasets are: {valid_datasets}")

    if dataset_name == "crestereo":
        return CREStereo(root=dataset_root, transforms=transforms)
    elif dataset_name == "carla-highres":
        return CarlaStereo(root=dataset_root, transforms=transforms)
    elif dataset_name == "instereo2k":
        return InStereo2k(root=dataset_root, transforms=transforms)
    elif dataset_name == "sintel":
        return SintelStereo(root=dataset_root, transforms=transforms)
    elif dataset_name == "sceneflow-monkaa":
        return SceneFlowStereo(root=dataset_root, transforms=transforms, split="monkaa", pass_name="both")
    elif dataset_name == "sceneflow-flyingthings":
        return SceneFlowStereo(root=dataset_root, transforms=transforms, split="flyingthings", pass_name="both")
    elif dataset_name == "sceneflow-driving":
        return SceneFlowStereo(root=dataset_root, transforms=transforms, split="driving", pass_name="both")
    elif dataset_name == "fallingthings":
        return FallingThingsStereo(root=dataset_root, transforms=transforms, variant="both")
    elif dataset_name == "eth3d-train":
        return ETH3DStereo(root=dataset_root, transforms=transforms, split="train")
    elif dataset_name == "eth3d-test":
        return ETH3DStereo(root=dataset_root, transforms=transforms, split="test")
    elif dataset_name == "kitti2015-train":
        return Kitti2015Stereo(root=dataset_root, transforms=transforms, split="train")
    elif dataset_name == "kitti2015-test":
        return Kitti2015Stereo(root=dataset_root, transforms=transforms, split="test")
    elif dataset_name == "kitti2012-train":
        return Kitti2012Stereo(root=dataset_root, transforms=transforms, split="train")
    elif dataset_name == "kitti2012-test":
        return Kitti2012Stereo(root=dataset_root, transforms=transforms, split="test")
    elif dataset_name == "middlebury2014-train-ambient":
        return Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="train", use_ambient_views=True, calibration="perfect")
    elif dataset_name == "middlebury2014-train":
        return Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="train", calibration="perfect")
    elif dataset_name == "middlebury2014-test":
        return Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="test", calibration="perfect")
    elif dataset_name == "middlebury2014-other":
        return Middlebury2014Stereo(root=dataset_root, transforms=transforms, split="additional", use_ambient_views=True, calibration="both")
        