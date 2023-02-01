# type: ignore

from __future__ import annotations

import contextlib

import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F

__all__ = ["wrap_dataset_for_transforms_v2"]

cache = functools.partial(functools.lru_cache, max_size=None)

_WRAPPERS = {}


# TODO: naming!
def wrap_dataset_for_transforms_v2(dataset: datasets.VisionDataset) -> _VisionDatasetDatapointWrapper:
    dataset_cls = type(dataset)
    wrapper = _WRAPPERS.get(dataset_cls)
    if wrapper is None:
        # TODO: If we have documentation on how to do that, put a link in the error message.
        msg = f"No wrapper exist for dataset class {dataset_cls.__name__}. Please wrap the output yourself."
        if dataset_cls in datasets.__dict__.values():
            msg = (
                f"{msg} If an automated wrapper for this dataset would be useful for you, "
                f"please open an issue at https://github.com/pytorch/vision/issues."
            )
        raise ValueError(msg)
    return _VisionDatasetDatapointWrapper(dataset, wrapper)


def raise_missing_functionality(dataset, *params):
    msg = f"{type(dataset).__name__}"
    if params:
        param_msg = ", ".join(f"{param}={getattr(dataset, param)}" for param in params)
        msg = f"{msg} with {param_msg}"
    raise RuntimeError(
        f"{msg} is currently not supported by this wrapper. "
        f"If this would be helpful for you, please open an issue at https://github.com/pytorch/vision/issues."
    )


class _VisionDatasetDatapointWrapper(Dataset):
    def __init__(self, dataset: datasets.VisionDataset, wrapper) -> None:
        self._vision_dataset = dataset
        self._wrapper = wrapper

        # We need to disable the transforms on the dataset here to be able to inject the wrapping before we apply the
        # transforms
        self.transforms, dataset.transforms = dataset.transforms, None

    def __getattr__(self, item: str) -> Any:
        with contextlib.suppress(AttributeError):
            return object.__getattribute__(self, item)

        return getattr(self._vision_dataset, item)

    def __getitem__(self, idx: int) -> Any:
        # This gets us the raw sample since we disabled the transforms for the underlying dataset in the constructor
        # of this class
        sample = self._vision_dataset[idx]

        sample = self._wrapper(self._vision_dataset, sample)

        # We don't need to care about `transform` and `target_transform` here since `VisionDataset` joins them into a
        # `transforms` internally:
        # https://github.com/pytorch/vision/blob/2d92728341bbd3dc1e0f1e86c6a436049bbb3403/torchvision/datasets/vision.py#L52-L54
        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample

    def __len__(self) -> int:
        return len(self._vision_dataset)


@cache
def get_categories(dataset: datasets.VisionDataset) -> Optional[List[str]]:
    categories_fn = {
        datasets.Caltech256: lambda dataset: [name.rsplit(".", 1)[1] for name in dataset.categories],
        datasets.CIFAR10: lambda dataset: dataset.classes,
        datasets.CIFAR100: lambda dataset: dataset.classes,
        datasets.FashionMNIST: lambda dataset: dataset.classes,
        datasets.ImageNet: lambda dataset: [", ".join(names) for names in dataset.classes],
    }.get(type(dataset))
    return categories_fn(dataset) if categories_fn is not None else None


def classification_wrapper(
    dataset: datasets.VisionDataset, sample: Tuple[PIL.Image.Image, Optional[int]]
) -> Tuple[PIL.Image.Image, Optional[datapoints.Label]]:
    image, label = sample
    if label is not None:
        label = datapoints.Label(label, categories=get_categories(dataset))
    return image, label


for dataset_type in [
    datasets.Caltech256,
    datasets.CIFAR10,
    datasets.CIFAR100,
    datasets.ImageNet,
    datasets.MNIST,
    datasets.FashionMNIST,
]:
    _WRAPPERS[dataset_type] = classification_wrapper


def segmentation_wrapper(
    dataset: datasets.VisionDataset, sample: Tuple[PIL.Image.Image, PIL.Image.Image]
) -> Tuple[PIL.Image.Image, datapoints.Mask]:
    image, mask = sample
    return image, datapoints.Mask(F.to_image_tensor(mask))


for dataset_type in [
    datasets.VOCSegmentation,
]:
    _WRAPPERS[dataset_type] = segmentation_wrapper


def caltech101_wrapper(
    dataset: datasets.Caltech101, sample: Tuple[PIL.Image.Image, Any]
) -> Tuple[PIL.Image.Image, Any]:
    if "annotation" in dataset.target_type:
        raise_missing_functionality(dataset, "target_type")

    image, target = sample

    return image, datapoints.Label(target, categories=dataset.categories)


_WRAPPERS[datasets.Caltech101] = caltech101_wrapper


@cache
def get_coco_detection_categories(dataset: datasets.CocoDetection):
    idx_to_category = {idx: cat["name"] for idx, cat in dataset.coco.cats.items()}
    idx_to_category[0] = "__background__"
    for idx in set(range(91)) - idx_to_category.keys():
        idx_to_category[idx] = "N/A"

    return [category for _, category in sorted(idx_to_category.items())]


def coco_dectection_wrapper(
    dataset: datasets.CocoDetection, sample: Tuple[PIL.Image.Image, List[Dict[str, Any]]]
) -> Tuple[PIL.Image.Image, Dict[str, List[Any]]]:
    def segmentation_to_mask(segmentation: Any, *, spatial_size: Tuple[int, int]) -> torch.Tensor:
        from pycocotools import mask

        segmentation = (
            mask.frPyObjects(segmentation, *spatial_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *spatial_size))
        )
        return torch.from_numpy(mask.decode(segmentation))

    image, target = sample

    # Originally, COCODetection returns a list of dicts in which each dict represents an object instance on the image.
    # However, our transforms and models expect all instance annotations grouped together, if applicable as tensor with
    # batch dimension. Thus, we are changing the target to a dict of lists here.
    batched_target = defaultdict(list)
    for object in target:
        for key, value in object.items():
            batched_target[key].append(value)

    spatial_size = tuple(F.get_spatial_size(image))
    batched_target = dict(
        batched_target,
        boxes=datapoints.BoundingBox(
            batched_target["bbox"],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=spatial_size,
        ),
        masks=datapoints.Mask(
            torch.stack(
                [
                    segmentation_to_mask(segmentation, spatial_size=spatial_size)
                    for segmentation in batched_target["segmentation"]
                ]
            ),
        ),
        labels=datapoints.Label(batched_target["category_id"], categories=get_coco_detection_categories(dataset)),
    )

    return image, batched_target


_WRAPPERS[datasets.CocoDetection] = coco_dectection_wrapper
_WRAPPERS[datasets.CocoCaptions] = lambda sample: sample


def voc_detection_wrapper(
    dataset: datasets.VOCDetection, sample: Tuple[PIL.Image.Image, Any]
) -> Tuple[PIL.Image.Image, Any]:
    categories = [
        "__background__",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    categories_to_idx = dict(zip(categories, range(len(categories))))

    image, target = sample

    batched_instances = defaultdict(list)
    for object in target["annotation"]["object"]:
        for key, value in object.items():
            batched_instances[key].append(value)

    target["boxes"] = datapoints.BoundingBox(
        [[int(bndbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")] for bndbox in batched_instances["bndbox"]],
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=tuple(int(target["annotation"]["size"][dim]) for dim in ("height", "width")),
    )
    target["labels"] = datapoints.Label(
        [categories_to_idx[category] for category in batched_instances["name"]],
        categories=categories,
    )

    return image, target


_WRAPPERS[datasets.VOCDetection] = voc_detection_wrapper


def sbd_wrapper(dataset: datasets.SBDataset, sample: Tuple[PIL.Image.Image, Any]) -> Tuple[PIL.Image.Image, Any]:
    if dataset.mode == "boundaries":
        raise_missing_functionality(dataset, "mode")

    image, target = sample
    return image, datapoints.Mask(F.to_image_tensor(target))


_WRAPPERS[datasets.SBDataset] = sbd_wrapper
