from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Callable, cast, Dict, Optional, Tuple, Type

import PIL.Image
import torch
from torch.utils._pytree import _get_node_type, LeafSpec, SUPPORTED_NODES, tree_flatten, tree_unflatten

from torchvision import datasets
from torchvision.prototype import features
from torchvision.transforms import functional as F


def tree_flatten_to_spec(pytree, spec):
    if isinstance(spec, LeafSpec):
        return [pytree]

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    flat = []
    for children_pytree, children_spec in zip(child_pytrees, spec.children_specs):
        flat.extend(tree_flatten_to_spec(children_pytree, children_spec))

    return flat


# FIXME: make this a proper dataset
class DatasetFeatureWrapper:
    __wrappers_fns__: Dict[
        Type[datasets.VisionDataset],
        Callable[[datasets.VisionDataset, bool, Dict[Type[features._Feature], Optional[torch.dtype]]], Any],
    ] = {}

    def __init__(self, dataset, wrappers):
        self.__dataset__ = dataset
        self.__wrappers__ = wrappers

    # FIXME: re-route everything to __dataset__ besides __getitem__

    def __getitem__(self, idx: int) -> Any:
        # Do we wrap before or after the transforms? -> most likely after
        sample = self.__dataset__[idx]

        wrappers_flat, spec = tree_flatten(self.__wrappers__)
        sample_flat = tree_flatten_to_spec(sample, spec)

        wrapped_sample_flat = [wrapper(item) for item, wrapper in zip(sample_flat, wrappers_flat)]

        return tree_unflatten(wrapped_sample_flat, spec)

    @classmethod
    def __register_wrappers_fn__(cls, dataset_type: Type[datasets.VisionDataset]):
        def foo(wrappers_fn):
            cls.__wrappers_fns__[dataset_type] = wrappers_fn
            return wrappers_fn

        return foo

    @classmethod
    def from_torchvision_dataset(
        cls, dataset: datasets.VisionDataset, *, keep_pil_image: bool = False, dtypes=None
    ) -> DatasetFeatureWrapper:
        dtypes = defaultdict(lambda: None, dtypes or dict())
        wrappers_fn = cls.__wrappers_fns__[type(dataset)]
        wrappers = wrappers_fn(dataset, keep_pil_image, dtypes)
        return cls(dataset, wrappers)


def identity_wrapper(obj):
    return obj


def generic_feature_wrapper(data):
    return features.GenericFeature(data)


def wrap_image(image, *, keep_pil_image, dtype):
    assert isinstance(image, PIL.Image.Image)
    if keep_pil_image:
        return image

    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image, dtype=dtype)
    return features.Image(image)


def make_image_wrapper(*, keep_pil_image, dtypes):
    return functools.partial(wrap_image, keep_pil_image=keep_pil_image, dtype=dtypes[features.Image])


def wrap_label(label, *, categories, dtype):
    return features.Label(label, categories=categories, dtype=dtype)


def make_label_wrapper(*, categories, dtypes):
    return functools.partial(wrap_label, categories=categories, dtype=dtypes[features.Label])


def wrap_segmentation_mask(segmentation_mask, *, dtype):
    assert isinstance(segmentation_mask, PIL.Image.Image)

    segmentation_mask = F.pil_to_tensor(segmentation_mask)
    segmentation_mask = F.convert_image_dtype(segmentation_mask, dtype=dtype)
    return features.Mask(segmentation_mask.squeeze(0))


def make_segmentation_mask_wrapper(*, dtypes):
    return functools.partial(wrap_segmentation_mask, dtype=dtypes[features.Mask])


CATEGORIES_GETTER = defaultdict(
    lambda: lambda dataset: None,
    {
        datasets.Caltech256: lambda dataset: [name.rsplit(".", 1)[1] for name in dataset.categories],
        datasets.CIFAR10: lambda dataset: dataset.classes,
        datasets.CIFAR100: lambda dataset: dataset.classes,
        datasets.FashionMNIST: lambda dataset: dataset.classes,
        datasets.ImageNet: lambda dataset: [", ".join(names) for names in dataset.classes],
    },
)


def classification_wrappers(dataset, keep_pil_image, dtypes):
    return (
        make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes),
        make_label_wrapper(categories=CATEGORIES_GETTER[type(dataset)](dataset), dtypes=dtypes),
    )


for dataset_type in [
    datasets.Caltech256,
    datasets.CIFAR10,
    datasets.CIFAR100,
    datasets.ImageNet,
    datasets.MNIST,
    datasets.FashionMNIST,
]:
    DatasetFeatureWrapper.__register_wrappers_fn__(dataset_type)(classification_wrappers)


def segmentation_wrappers(dataset, keep_pil_image, dtypes):
    return (
        make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes),
        make_segmentation_mask_wrapper(dtypes=dtypes),
    )


for dataset_type in [
    datasets.VOCSegmentation,
]:
    DatasetFeatureWrapper.__register_wrappers_fn__(dataset_type)(classification_wrappers)


@DatasetFeatureWrapper.__register_wrappers_fn__(datasets.Caltech101)
def caltech101_dectection_wrappers(dataset, keep_pil_image, dtypes):
    target_type_wrapper_map = {
        "category": make_label_wrapper(categories=dataset.categories, dtypes=dtypes),
        "annotation": features.GenericFeature,
    }
    return (
        make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes),
        [target_type_wrapper_map[target_type] for target_type in dataset.target_type],
    )


@DatasetFeatureWrapper.__register_wrappers_fn__(datasets.CocoDetection)
def coco_dectection_wrappers(dataset, keep_pil_image, dtypes):
    idx_to_category = {idx: cat["name"] for idx, cat in dataset.coco.cats.items()}
    idx_to_category[0] = "__background__"
    for idx in set(range(91)) - idx_to_category.keys():
        idx_to_category[idx] = "N/A"

    categories = [category for _, category in sorted(idx_to_category.items())]

    def segmentation_to_mask(segmentation: Any, *, iscrowd: bool, image_size: Tuple[int, int]) -> torch.Tensor:
        from pycocotools import mask

        segmentation = (
            mask.frPyObjects(segmentation, *image_size)
            if iscrowd
            else mask.merge(mask.frPyObjects(segmentation, *image_size))
        )
        return torch.from_numpy(mask.decode(segmentation)).to(torch.bool)

    def sample_wrapper(sample):
        image, target = sample

        _, height, width = F.get_dimensions(image)
        image_size = height, width

        wrapped_image = wrap_image(image, keep_pil_image=keep_pil_image, dtype=dtypes[features.Image])

        batched_target = defaultdict(list)
        for object in target:
            for key, value in object.items():
                batched_target[key].append(value)

        wrapped_target = dict(
            batched_target,
            segmentation=features.Mask(
                torch.stack(
                    [
                        segmentation_to_mask(segmentation, iscrowd=iscrowd, image_size=image_size)
                        for segmentation, iscrowd in zip(batched_target["segmentation"], batched_target["iscrowd"])
                    ]
                ),
                dtype=dtypes.get(features.Mask),
            ),
            bbox=features.BoundingBox(
                batched_target["bbox"],
                format=features.BoundingBoxFormat.XYXY,
                image_size=image_size,
                dtype=dtypes.get(features.BoundingBox),
            ),
            labels=features.Label(batched_target.pop("category_id"), categories=categories),
        )

        return wrapped_image, wrapped_target

    return sample_wrapper


@DatasetFeatureWrapper.__register_wrappers_fn__(datasets.CocoCaptions)
def coco_captions_wrappers(dataset, keep_pil_image, dtypes):
    return make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), identity_wrapper


@DatasetFeatureWrapper.__register_wrappers_fn__(datasets.VOCDetection)
def voc_detection_wrappers(dataset, keep_pil_image, dtypes):
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

    def target_wrapper(target):
        batched_object = defaultdict(list)
        for object in target["annotation"]["object"]:
            for key, value in object.items():
                batched_object[key].append(value)

        wrapped_object = dict(
            batched_object,
            bndbox=features.BoundingBox(
                [
                    [int(bndbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                    for bndbox in batched_object["bndbox"]
                ],
                format="xyxy",
                image_size=cast(
                    Tuple[int, int], tuple(int(target["annotation"]["size"][dim]) for dim in ("height", "width"))
                ),
            ),
        )
        wrapped_object["labels"] = features.Label(
            [categories_to_idx[category] for category in batched_object["name"]],
            categories=categories,
            dtype=dtypes[features.Label],
        )

        target["annotation"]["object"] = wrapped_object
        return target

    return make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), target_wrapper


@DatasetFeatureWrapper.__register_wrappers_fn__(datasets.SBDataset)
def sbd_wrappers(dataset, keep_pil_image, dtypes):
    return {
        "boundaries": (make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), generic_feature_wrapper),
        "segmentation": segmentation_wrappers(dataset, keep_pil_image, dtypes),
    }[dataset.mode]
