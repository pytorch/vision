# type: ignore

from __future__ import annotations

import contextlib

import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F

T = TypeVar("T")
D = TypeVar("D", bound=datasets.VisionDataset)

__all__ = ["wrap_dataset_for_transforms_v2"]

_WRAPPERS = {}


# TODO: naming!
def wrap_dataset_for_transforms_v2(dataset: datasets.VisionDataset) -> _VisionDatasetDatapointWrapper:
    wrapper = _WRAPPERS.get(type(dataset))
    if wrapper is None:
        raise TypeError
    return _VisionDatasetDatapointWrapper(dataset, wrapper)


class _VisionDatasetDatapointWrapper(Dataset):
    def __init__(self, dataset: datasets.VisionDataset, wrapper) -> None:
        self.vision_dataset = dataset
        self.wrapper = wrapper

        # We need to disable the transforms on the dataset here to be able to inject the wrapping before we apply the
        # transforms
        self.transform, dataset.transform = dataset.transform, None
        self.target_transform, dataset.target_transform = dataset.target_transform, None
        self.transforms, dataset.transforms = dataset.transforms, None

    def __getattr__(self, item: str) -> Any:
        with contextlib.suppress(AttributeError):
            return object.__getattribute__(self, item)

        return getattr(self.vision_dataset, item)

    def __getitem__(self, idx: int) -> Any:
        # This gets us the raw sample since we disabled the transforms for the underlying dataset in the constructor
        # of this class
        sample = self.vision_dataset[idx]

        sample = self.wrapper(self.vision_dataset, sample)

        # We don't need to care about `transform` and `target_transform` here since `VisionDataset` joins them into a
        # `transforms` internally:
        # https://github.com/pytorch/vision/blob/2d92728341bbd3dc1e0f1e86c6a436049bbb3403/torchvision/datasets/vision.py#L52-L54
        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample

    def __len__(self) -> int:
        return len(self.vision_dataset)


def identity_wrapper(sample: T) -> T:
    return sample


@functools.lru_cache(maxsize=None)
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
    image, target = sample

    target_type_wrapper_map: Dict[str, Callable] = {
        "category": lambda label: datapoints.Label(label, categories=dataset.categories),
        "annotation": datapoints.GenericDatapoint,
    }
    if len(dataset.target_type) == 1:
        target = target_type_wrapper_map[dataset.target_type[0]](target)
    else:
        target = tuple(target_type_wrapper_map[typ](item) for typ, item in zip(dataset.target_type, target))

    return image, target


_WRAPPERS[datasets.Caltech101] = caltech101_wrapper


def coco_dectection_wrapper(
    dataset: datasets.CocoDetection, sample: Tuple[PIL.Image.Image, List[Dict[str, Any]]]
) -> Tuple[PIL.Image.Image, Dict[str, List[Any]]]:
    idx_to_category = {idx: cat["name"] for idx, cat in dataset.coco.cats.items()}
    idx_to_category[0] = "__background__"
    for idx in set(range(91)) - idx_to_category.keys():
        idx_to_category[idx] = "N/A"

    categories = [category for _, category in sorted(idx_to_category.items())]

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
        labels=datapoints.Label(batched_target["category_id"], categories=categories),
    )

    return image, batched_target


_WRAPPERS[datasets.CocoDetection] = coco_dectection_wrapper
_WRAPPERS[datasets.CocoCaptions] = identity_wrapper


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
    image, target = sample

    if dataset.mode == "boundaries":
        target = datapoints.GenericDatapoint(target)
    else:
        target = datapoints.Mask(F.to_image_tensor(target))

    return image, target


_WRAPPERS[datasets.SBDataset] = sbd_wrapper
