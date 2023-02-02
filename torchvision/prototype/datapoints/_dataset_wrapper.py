# type: ignore

from __future__ import annotations

import contextlib
import functools

import torch
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision._utils import sequence_to_str
from torchvision.prototype import datapoints
from torchvision.prototype.transforms import functional as F

__all__ = ["wrap_dataset_for_transforms_v2"]

cache = functools.partial(functools.lru_cache)

_WRAPPERS = {}


# TODO: naming!
def wrap_dataset_for_transforms_v2(dataset):
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


class _VisionDatasetDatapointWrapper(Dataset):
    def __init__(self, dataset, wrapper):
        self._vision_dataset = dataset
        self._wrapper = wrapper

        # We need to disable the transforms on the dataset here to be able to inject the wrapping before we apply the
        # transforms
        self.transforms, dataset.transforms = dataset.transforms, None

    def __getattr__(self, item):
        with contextlib.suppress(AttributeError):
            return object.__getattribute__(self, item)

        return getattr(self._vision_dataset, item)

    def __getitem__(self, idx):
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

    def __len__(self):
        return len(self._vision_dataset)


def identity(item):
    return item


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    if not list_of_dicts:
        return {}

    dict_of_lists = {key: [value] for key, value in list_of_dicts[0].items()}
    for dct in list_of_dicts[1:]:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict_of_lists


def wrap_target_by_type(dataset, target, type_wrappers, *, fail_on=(), attr_name="target_type"):
    if target is None:
        return None

    target_types = getattr(dataset, attr_name)

    if any(target_type in fail_on for target_type in target_types):
        raise RuntimeError(
            f"{type(dataset).__name__} with target type(s) {sequence_to_str(fail_on, separate_last='or ')} "
            f"is currently not supported by this wrapper. "
            f"If this would be helpful for you, please open an issue at https://github.com/pytorch/vision/issues."
        )

    if not isinstance(target, (tuple, list)):
        target = [target]

    wrapped_target = tuple(
        type_wrappers.get(target_type, identity)(item) for target_type, item in zip(target_types, target)
    )

    if len(wrapped_target) == 1:
        wrapped_target = wrapped_target[0]

    return wrapped_target


@cache
def get_categories(dataset):
    categories_fn = {
        datasets.Caltech256: lambda dataset: [name.rsplit(".", 1)[1] for name in dataset.categories],
        datasets.CIFAR10: lambda dataset: dataset.classes,
        datasets.CIFAR100: lambda dataset: dataset.classes,
        datasets.FashionMNIST: lambda dataset: dataset.classes,
        datasets.ImageNet: lambda dataset: [", ".join(names) for names in dataset.classes],
        datasets.DatasetFolder: lambda dataset: dataset.classes,
        datasets.ImageFolder: lambda dataset: dataset.classes,
    }.get(type(dataset))
    return categories_fn(dataset) if categories_fn is not None else None


def classification_wrapper(dataset, sample):
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
    datasets.GTSRB,
    datasets.DatasetFolder,
    datasets.ImageFolder,
]:
    _WRAPPERS[dataset_type] = classification_wrapper


def segmentation_wrapper(dataset, sample):
    image, mask = sample
    return image, datapoints.Mask(F.to_image_tensor(mask).squeeze(0))


for dataset_type in [
    datasets.VOCSegmentation,
]:
    _WRAPPERS[dataset_type] = segmentation_wrapper


def video_classification_wrapper(dataset, sample):
    if dataset.video_clips.output_format == "THWC":
        raise RuntimeError(
            f"{type(dataset).__name__} with output_format='THWC' is not supported by this wrapper, "
            f"since it is not compatible with the transformations. Please use output_format='TCHW' instead."
        )

    video, audio, label = sample

    video = datapoints.Video(video)
    label = datapoints.Label(label, categories=dataset.classes)

    return video, audio, label


for dataset_type in [
    datasets.HMDB51,
    datasets.Kinetics,
    datasets.UCF101,
]:
    _WRAPPERS[dataset_type] = video_classification_wrapper


def caltech101_wrapper(dataset, sample):
    image, target = sample
    return image, wrap_target_by_type(
        dataset,
        target,
        {"category": lambda item: datapoints.Label(target, categories=dataset.categories)},
        fail_on=["annotation"],
    )


_WRAPPERS[datasets.Caltech101] = caltech101_wrapper


@cache
def get_coco_detection_categories(dataset):
    idx_to_category = {idx: cat["name"] for idx, cat in dataset.coco.cats.items()}
    idx_to_category[0] = "__background__"
    for idx in set(range(91)) - idx_to_category.keys():
        idx_to_category[idx] = "N/A"

    return [category for _, category in sorted(idx_to_category.items())]


def coco_dectection_wrapper(dataset, sample):
    def segmentation_to_mask(segmentation, *, spatial_size):
        from pycocotools import mask

        segmentation = (
            mask.frPyObjects(segmentation, *spatial_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *spatial_size))
        )
        return torch.from_numpy(mask.decode(segmentation))

    image, target = sample

    batched_target = list_of_dicts_to_dict_of_lists(target)

    spatial_size = tuple(F.get_spatial_size(image))
    batched_target["boxes"] = datapoints.BoundingBox(
        batched_target["bbox"],
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=spatial_size,
    )
    batched_target["masks"] = datapoints.Mask(
        torch.stack(
            [
                segmentation_to_mask(segmentation, spatial_size=spatial_size)
                for segmentation in batched_target["segmentation"]
            ]
        ),
    )
    batched_target["labels"] = datapoints.Label(
        batched_target["category_id"], categories=get_coco_detection_categories(dataset)
    )

    return image, batched_target


_WRAPPERS[datasets.CocoDetection] = coco_dectection_wrapper
_WRAPPERS[datasets.CocoCaptions] = identity


VOC_DETECTION_CATEGORIES = [
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
VOC_DETECTION_CATEGORY_TO_IDX = dict(zip(VOC_DETECTION_CATEGORIES, range(len(VOC_DETECTION_CATEGORIES))))


def voc_detection_wrapper(dataset, sample):
    image, target = sample

    batched_instances = list_of_dicts_to_dict_of_lists(target["annotation"]["object"])

    target["boxes"] = datapoints.BoundingBox(
        [[int(bndbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")] for bndbox in batched_instances["bndbox"]],
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=(image.height, image.width),
    )
    target["labels"] = datapoints.Label(
        [VOC_DETECTION_CATEGORY_TO_IDX[category] for category in batched_instances["name"]],
        categories=VOC_DETECTION_CATEGORIES,
    )

    return image, target


_WRAPPERS[datasets.VOCDetection] = voc_detection_wrapper


def sbd_wrapper(dataset, sample):
    if dataset.mode == "boundaries":
        raise RuntimeError(
            "SBDataset with mode='boundaries' is currently not supported by this wrapper. "
            "If this would be helpful for you, please open an issue at https://github.com/pytorch/vision/issues."
        )

    image, target = sample
    return image, datapoints.Mask(F.to_image_tensor(target).squeeze(0))


_WRAPPERS[datasets.SBDataset] = sbd_wrapper


def celeba_wrapper(dataset, sample):
    image, target = sample
    return wrap_target_by_type(
        dataset,
        target,
        {
            "identity": datapoints.Label,
            "bbox": lambda item: datapoints.BoundingBox(
                item, format=datapoints.BoundingBoxFormat.XYWH, spatial_size=(image.height, image.width)
            ),
        },
        # FIXME: Failing on "attr" here is problematic, since it is the default
        fail_on=["attr", "landmarks"],
    )


_WRAPPERS[datasets.CelebA] = celeba_wrapper

KITTI_CATEGORIES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
KITTI_CATEGORY_TO_IDX = dict(zip(KITTI_CATEGORIES, range(len(KITTI_CATEGORIES))))


def kitti_wrapper(dataset, sample):
    image, target = sample

    target = list_of_dicts_to_dict_of_lists(target)

    target["boxes"] = datapoints.BoundingBox(
        target["bbox"], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(image.height, image.width)
    )
    target["labels"] = datapoints.Label(
        [KITTI_CATEGORY_TO_IDX[category] for category in target["type"]], categories=KITTI_CATEGORIES
    )

    return image, target


_WRAPPERS[datasets.Kitti] = kitti_wrapper


def oxford_iiit_pet_wrapper(dataset, sample):
    image, target = sample
    return image, wrap_target_by_type(
        dataset,
        target,
        {
            "category": lambda item: datapoints.Label(item, categories=dataset.classes),
            "segmentation": lambda item: datapoints.Mask(F.pil_to_tensor(item).squeeze(0)),
        },
        attr_name="_target_types",
    )


_WRAPPERS[datasets.OxfordIIITPet] = oxford_iiit_pet_wrapper


def cityscapes_wrapper(dataset, sample):
    def instance_segmentation_wrapper(mask):
        # See https://github.com/mcordts/cityscapesScripts/blob/8da5dd00c9069058ccc134654116aac52d4f6fa2/cityscapesscripts/preparation/json2instanceImg.py#L7-L21
        data = F.pil_to_tensor(mask).squeeze(0)
        masks = []
        labels = []
        for id in data.unique():
            masks.append(data == id)
            label = id
            if label >= 1_000:
                label //= 1_000
            labels.append(label)
        masks = datapoints.Mask(torch.stack(masks))
        # FIXME: without the labels, returning just the instance masks is pretty useless. However, we would need to
        #  return a two-tuple or the like where we originally only had a single PIL image.
        labels = datapoints.Label(torch.stack(labels), categories=[cls.name for cls in dataset.classes])
        return masks

    image, target = sample
    return image, wrap_target_by_type(
        dataset,
        target,
        {
            "instance": instance_segmentation_wrapper,
            "semantic": lambda item: datapoints.Mask(F.pil_to_tensor(item).squeeze(0)),
        },
        fail_on=["polygon", "color"],
    )


_WRAPPERS[datasets.Cityscapes] = cityscapes_wrapper


def widerface_wrapper(dataset, sample):
    image, target = sample
    if target is not None:
        # FIXME: all returned values inside this dictionary are tensors, but not images
        target["bbox"] = datapoints.BoundingBox(
            target["bbox"], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=(image.height, image.width)
        )
    return image, target


_WRAPPERS[datasets.WIDERFace] = widerface_wrapper
