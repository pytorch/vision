from __future__ import annotations

import contextlib

import functools
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import PIL.Image
import torch
from torch.utils._pytree import (
    _get_node_type,
    LeafSpec,
    PyTree,
    SUPPORTED_NODES,
    tree_flatten,
    tree_unflatten,
    TreeSpec,
)
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision.prototype import datapoints
from torchvision.transforms import functional as F

T = TypeVar("T")


def tree_flatten_to_spec(pytree: PyTree, spec: TreeSpec) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    flat = []
    for children_pytree, children_spec in zip(child_pytrees, spec.children_specs):
        flat.extend(tree_flatten_to_spec(children_pytree, children_spec))

    return flat


class VisionDatasetFeatureWrapper(Dataset):
    _wrappers_fns: Dict[Type[datasets.VisionDataset], PyTree] = {}

    def __init__(self, dataset: datasets.VisionDataset, wrappers: PyTree) -> None:
        self.vision_dataset = dataset
        self.sample_wrappers = wrappers

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

        wrappers_flat, spec = tree_flatten(self.sample_wrappers)
        # We cannot use `tree_flatten` directly, because the spec of `self.sample_wrappers` and `sample` might differ.
        # For example, for `COCODetection` the target is a list of dicts. To be able to wrap this into a dict of lists,
        # we need to have access to the whole list, but `tree_flatten` would also flatten it.
        sample_flat = tree_flatten_to_spec(sample, spec)
        wrapped_sample_flat = [wrapper(item) for item, wrapper in zip(sample_flat, wrappers_flat)]
        sample = tree_unflatten(wrapped_sample_flat, spec)

        # We don't need to care about `transform` and `target_transform` here since `VisionDataset` joins them into a
        # `transforms` internally:
        # https://github.com/pytorch/vision/blob/2d92728341bbd3dc1e0f1e86c6a436049bbb3403/torchvision/datasets/vision.py#L52-L54
        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample

    def __len__(self) -> int:
        return len(self.vision_dataset)

    @classmethod
    def _register_wrappers_fn(cls, dataset_type: Type[datasets.VisionDataset]) -> Callable[[PyTree], PyTree]:
        def register(wrappers_fn: PyTree) -> PyTree:
            cls._wrappers_fns[dataset_type] = wrappers_fn
            return wrappers_fn

        return register

    @classmethod
    def from_torchvision_dataset(
        cls,
        dataset: datasets.VisionDataset,
        *,
        keep_pil_image: bool = False,
        bounding_box_format: Optional[datapoints.BoundingBoxFormat] = None,
        dtypes: Optional[Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]]] = None,
    ) -> VisionDatasetFeatureWrapper:
        dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]] = {
            datapoints.Image: torch.uint8,
            datapoints.Label: torch.int64,
            datapoints.Mask: torch.uint8,
            datapoints.BoundingBox: torch.float32,
            datapoints.GenericDatapoint: None,
            **(dtypes or dict()),
        }
        wrappers_fn = cls._wrappers_fns[type(dataset)]
        wrappers = wrappers_fn(dataset, keep_pil_image, bounding_box_format, dtypes)
        return cls(dataset, wrappers)


def identity_wrapper(obj: T) -> T:
    return obj


def generic_feature_wrapper(data: Any) -> datapoints.GenericDatapoint:
    return datapoints.GenericDatapoint(data)


def wrap_image(image: PIL.Image.Image, *, dtype: Optional[torch.dtype]) -> datapoints.Image:
    image = F.pil_to_tensor(image)
    if dtype is not None:
        image = F.convert_image_dtype(image, dtype=dtype)
    return datapoints.Image(image)


def make_image_wrapper(
    *, keep_pil_image: bool, dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]]
) -> Callable[[PIL.Image.Image], Union[PIL.Image.Image, datapoints.Image]]:
    if keep_pil_image:
        return identity_wrapper

    return functools.partial(wrap_image, dtype=dtypes[datapoints.Image])


def wrap_label(label: Any, *, categories: Optional[Sequence[str]], dtype: Optional[torch.dtype]) -> datapoints.Label:
    return datapoints.Label(label, categories=categories, dtype=dtype)


def make_label_wrapper(
    *, categories: Optional[Sequence[str]], dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]]
) -> Callable[[Any], datapoints.Label]:
    return functools.partial(wrap_label, categories=categories, dtype=dtypes[datapoints.Label])


def wrap_segmentation_mask(segmentation_mask: PIL.Image.Image, *, dtype: Optional[torch.dtype]) -> datapoints.Mask:
    assert isinstance(segmentation_mask, PIL.Image.Image)

    segmentation_mask = F.pil_to_tensor(segmentation_mask)
    if dtype is not None:
        segmentation_mask = F.convert_image_dtype(segmentation_mask, dtype=dtype)
    return datapoints.Mask(segmentation_mask.squeeze(0))


def make_segmentation_mask_wrapper(
    *, dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]]
) -> Callable[[Any], datapoints.Mask]:
    return functools.partial(wrap_segmentation_mask, dtype=dtypes[datapoints.Mask])


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


def classification_wrappers(
    dataset: datasets.VisionDataset,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
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
    VisionDatasetFeatureWrapper._register_wrappers_fn(dataset_type)(classification_wrappers)


def segmentation_wrappers(
    dataset: datasets.VisionDataset,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
    return (
        make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes),
        make_segmentation_mask_wrapper(dtypes=dtypes),
    )


for dataset_type in [
    datasets.VOCSegmentation,
]:
    VisionDatasetFeatureWrapper._register_wrappers_fn(dataset_type)(segmentation_wrappers)


@VisionDatasetFeatureWrapper._register_wrappers_fn(datasets.Caltech101)
def caltech101_dectection_wrappers(
    dataset: datasets.Caltech101,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
    target_type_wrapper_map = {
        "category": make_label_wrapper(categories=dataset.categories, dtypes=dtypes),
        "annotation": datapoints.GenericDatapoint,
    }
    return (
        make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes),
        [target_type_wrapper_map[target_type] for target_type in dataset.target_type],
    )


@VisionDatasetFeatureWrapper._register_wrappers_fn(datasets.CocoDetection)
def coco_dectection_wrappers(
    dataset: datasets.CocoDetection,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
    idx_to_category = {idx: cat["name"] for idx, cat in dataset.coco.cats.items()}
    idx_to_category[0] = "__background__"
    for idx in set(range(91)) - idx_to_category.keys():
        idx_to_category[idx] = "N/A"

    categories = [category for _, category in sorted(idx_to_category.items())]

    def segmentation_to_mask(segmentation: Any, *, iscrowd: bool, spatial_size: Tuple[int, int]) -> torch.Tensor:
        from pycocotools import mask

        segmentation = (
            mask.frPyObjects(segmentation, *spatial_size)
            if iscrowd
            else mask.merge(mask.frPyObjects(segmentation, *spatial_size))
        )
        return torch.from_numpy(mask.decode(segmentation)).to(torch.bool)

    def sample_wrapper(sample: Tuple[PIL.Image, List[Dict[str, Any]]]) -> Tuple[datapoints.Image, Dict[str, Any]]:
        image, target = sample

        _, height, width = F.get_dimensions(image)
        spatial_size = height, width

        image_wrapper = make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes)
        wrapped_image = image_wrapper(image)

        batched_target = defaultdict(list)
        for object in target:
            for key, value in object.items():
                batched_target[key].append(value)

        wrapped_target = dict(
            batched_target,
            segmentation=datapoints.Mask(
                torch.stack(
                    [
                        segmentation_to_mask(segmentation, iscrowd=iscrowd, spatial_size=spatial_size)
                        for segmentation, iscrowd in zip(batched_target["segmentation"], batched_target["iscrowd"])
                    ]
                ),
                dtype=dtypes.get(datapoints.Mask),
            ),
            bbox=datapoints.BoundingBox(
                batched_target["bbox"],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=spatial_size,
                dtype=dtypes.get(datapoints.BoundingBox),
            ),
            labels=datapoints.Label(batched_target.pop("category_id"), categories=categories),
        )
        if bounding_box_format is not None:
            wrapped_target["bbox"] = cast(datapoints.BoundingBox, wrapped_target["bbox"]).to_format(bounding_box_format)

        return wrapped_image, wrapped_target

    return sample_wrapper


@VisionDatasetFeatureWrapper._register_wrappers_fn(datasets.CocoCaptions)
def coco_captions_wrappers(
    dataset: datasets.CocoCaptions,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
    return make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), identity_wrapper


@VisionDatasetFeatureWrapper._register_wrappers_fn(datasets.VOCDetection)
def voc_detection_wrappers(
    dataset: datasets.VOCDetection,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
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

    def target_wrapper(target: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        batched_object = defaultdict(list)
        for object in target["annotation"]["object"]:
            for key, value in object.items():
                batched_object[key].append(value)

        wrapped_object = dict(
            batched_object,
            bndbox=datapoints.BoundingBox(
                [
                    [int(bndbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                    for bndbox in batched_object["bndbox"]
                ],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=cast(
                    Tuple[int, int], tuple(int(target["annotation"]["size"][dim]) for dim in ("height", "width"))
                ),
            ),
        )
        if bounding_box_format is not None:
            wrapped_object["bndbox"] = cast(datapoints.BoundingBox, wrapped_object["bndbox"]).to_format(
                bounding_box_format
            )
        wrapped_object["labels"] = datapoints.Label(
            [categories_to_idx[category] for category in batched_object["name"]],
            categories=categories,
            dtype=dtypes[datapoints.Label],
        )

        target["annotation"]["object"] = wrapped_object
        return target

    return make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), target_wrapper


@VisionDatasetFeatureWrapper._register_wrappers_fn(datasets.SBDataset)
def sbd_wrappers(
    dataset: datasets.SBDataset,
    keep_pil_image: bool,
    bounding_box_format: Optional[datapoints.BoundingBoxFormat],
    dtypes: Dict[Type[datapoints._datapoint.Datapoint], Optional[torch.dtype]],
) -> Any:
    return {
        "boundaries": (make_image_wrapper(keep_pil_image=keep_pil_image, dtypes=dtypes), generic_feature_wrapper),
        "segmentation": segmentation_wrappers(dataset, keep_pil_image, bounding_box_format, dtypes),
    }[dataset.mode]
