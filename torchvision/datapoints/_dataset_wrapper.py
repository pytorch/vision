# type: ignore

from __future__ import annotations

import contextlib
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from torchvision import datapoints, datasets
from torchvision.transforms.v2 import functional as F

__all__ = ["wrap_dataset_for_transforms_v2"]


def wrap_dataset_for_transforms_v2(dataset):
    """[BETA] Wrap a ``torchvision.dataset`` for usage with :mod:`torchvision.transforms.v2`.

    .. v2betastatus:: wrap_dataset_for_transforms_v2 function

    Example:
        >>> dataset = torchvision.datasets.CocoDetection(...)
        >>> dataset = wrap_dataset_for_transforms_v2(dataset)

    .. note::

       For now, only the most popular datasets are supported. Furthermore, the wrapper only supports dataset
       configurations that are fully supported by ``torchvision.transforms.v2``. If you encounter an error prompting you
       to raise an issue to ``torchvision`` for a dataset or configuration that you need, please do so.

    The dataset samples are wrapped according to the description below.

    Special cases:

        * :class:`~torchvision.datasets.CocoDetection`: Instead of returning the target as list of dicts, the wrapper
          returns a dict of lists. In addition, the key-value-pairs ``"boxes"`` (in ``XYXY`` coordinate format),
          ``"masks"`` and ``"labels"`` are added and wrap the data in the corresponding ``torchvision.datapoints``.
          The original keys are preserved.
        * :class:`~torchvision.datasets.VOCDetection`: The key-value-pairs ``"boxes"`` and ``"labels"`` are added to
          the target and wrap the data in the corresponding ``torchvision.datapoints``. The original keys are
          preserved.
        * :class:`~torchvision.datasets.CelebA`: The target for ``target_type="bbox"`` is converted to the ``XYXY``
          coordinate format and wrapped into a :class:`~torchvision.datapoints.BoundingBox` datapoint.
        * :class:`~torchvision.datasets.Kitti`: Instead returning the target as list of dictsthe wrapper returns a dict
          of lists. In addition, the key-value-pairs ``"boxes"`` and ``"labels"`` are added and wrap the data
          in the corresponding ``torchvision.datapoints``. The original keys are preserved.
        * :class:`~torchvision.datasets.OxfordIIITPet`: The target for ``target_type="segmentation"`` is wrapped into a
          :class:`~torchvision.datapoints.Mask` datapoint.
        * :class:`~torchvision.datasets.Cityscapes`: The target for ``target_type="semantic"`` is wrapped into a
          :class:`~torchvision.datapoints.Mask` datapoint. The target for ``target_type="instance"`` is *replaced* by
          a dictionary with the key-value-pairs ``"masks"`` (as :class:`~torchvision.datapoints.Mask` datapoint) and
          ``"labels"``.
        * :class:`~torchvision.datasets.WIDERFace`: The value for key ``"bbox"`` in the target is converted to ``XYXY``
          coordinate format and wrapped into a :class:`~torchvision.datapoints.BoundingBox` datapoint.

    Image classification datasets

        This wrapper is a no-op for image classification datasets, since they were already fully supported by
        :mod:`torchvision.transforms` and thus no change is needed for :mod:`torchvision.transforms.v2`.

    Segmentation datasets

        Segmentation datasets, e.g. :class:`~torchvision.datasets.VOCSegmentation` return a two-tuple of
        :class:`PIL.Image.Image`'s. This wrapper leaves the image as is (first item), while wrapping the
        segmentation mask into a :class:`~torchvision.datapoints.Mask` (second item).

    Video classification datasets

        Video classification datasets, e.g. :class:`~torchvision.datasets.Kinetics` return a three-tuple containing a
        :class:`torch.Tensor` for the video and audio and a :class:`int` as label. This wrapper wraps the video into a
        :class:`~torchvision.datapoints.Video` while leaving the other items as is.

        .. note::

            Only datasets constructed with ``output_format="TCHW"`` are supported, since the alternative
            ``output_format="THWC"`` is not supported by :mod:`torchvision.transforms.v2`.

    Args:
        dataset: the dataset instance to wrap for compatibility with transforms v2.
    """
    return VisionDatasetDatapointWrapper(dataset)


class WrapperFactories(dict):
    def register(self, dataset_cls):
        def decorator(wrapper_factory):
            self[dataset_cls] = wrapper_factory
            return wrapper_factory

        return decorator


# We need this two-stage design, i.e. a wrapper factory producing the actual wrapper, since some wrappers depend on the
# dataset instance rather than just the class, since they require the user defined instance attributes. Thus, we can
# provide a wrapping from the dataset class to the factory here, but can only instantiate the wrapper at runtime when
# we have access to the dataset instance.
WRAPPER_FACTORIES = WrapperFactories()


class VisionDatasetDatapointWrapper(Dataset):
    def __init__(self, dataset):
        dataset_cls = type(dataset)

        if not isinstance(dataset, datasets.VisionDataset):
            raise TypeError(
                f"This wrapper is meant for subclasses of `torchvision.datasets.VisionDataset`, "
                f"but got a '{dataset_cls.__name__}' instead."
            )

        for cls in dataset_cls.mro():
            if cls in WRAPPER_FACTORIES:
                wrapper_factory = WRAPPER_FACTORIES[cls]
                break
            elif cls is datasets.VisionDataset:
                # TODO: If we have documentation on how to do that, put a link in the error message.
                msg = f"No wrapper exists for dataset class {dataset_cls.__name__}. Please wrap the output yourself."
                if dataset_cls in datasets.__dict__.values():
                    msg = (
                        f"{msg} If an automated wrapper for this dataset would be useful for you, "
                        f"please open an issue at https://github.com/pytorch/vision/issues."
                    )
                raise TypeError(msg)

        self._dataset = dataset
        self._wrapper = wrapper_factory(dataset)

        # We need to disable the transforms on the dataset here to be able to inject the wrapping before we apply them.
        # Although internally, `datasets.VisionDataset` merges `transform` and `target_transform` into the joint
        # `transforms`
        # https://github.com/pytorch/vision/blob/135a0f9ea9841b6324b4fe8974e2543cbb95709a/torchvision/datasets/vision.py#L52-L54
        # some (if not most) datasets still use `transform` and `target_transform` individually. Thus, we need to
        # disable all three here to be able to extract the untransformed sample to wrap.
        self.transform, dataset.transform = dataset.transform, None
        self.target_transform, dataset.target_transform = dataset.target_transform, None
        self.transforms, dataset.transforms = dataset.transforms, None

    def __getattr__(self, item):
        with contextlib.suppress(AttributeError):
            return object.__getattribute__(self, item)

        return getattr(self._dataset, item)

    def __getitem__(self, idx):
        # This gets us the raw sample since we disabled the transforms for the underlying dataset in the constructor
        # of this class
        sample = self._dataset[idx]

        sample = self._wrapper(idx, sample)

        # Regardless of whether the user has supplied the transforms individually (`transform` and `target_transform`)
        # or joint (`transforms`), we can access the full functionality through `transforms`
        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample

    def __len__(self):
        return len(self._dataset)


def raise_not_supported(description):
    raise RuntimeError(
        f"{description} is currently not supported by this wrapper. "
        f"If this would be helpful for you, please open an issue at https://github.com/pytorch/vision/issues."
    )


def identity(item):
    return item


def identity_wrapper_factory(dataset):
    def wrapper(idx, sample):
        return sample

    return wrapper


def pil_image_to_mask(pil_image):
    return datapoints.Mask(pil_image)


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)


def wrap_target_by_type(target, *, target_types, type_wrappers):
    if not isinstance(target, (tuple, list)):
        target = [target]

    wrapped_target = tuple(
        type_wrappers.get(target_type, identity)(item) for target_type, item in zip(target_types, target)
    )

    if len(wrapped_target) == 1:
        wrapped_target = wrapped_target[0]

    return wrapped_target


def classification_wrapper_factory(dataset):
    return identity_wrapper_factory(dataset)


for dataset_cls in [
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
    WRAPPER_FACTORIES.register(dataset_cls)(classification_wrapper_factory)


def segmentation_wrapper_factory(dataset):
    def wrapper(idx, sample):
        image, mask = sample
        return image, pil_image_to_mask(mask)

    return wrapper


for dataset_cls in [
    datasets.VOCSegmentation,
]:
    WRAPPER_FACTORIES.register(dataset_cls)(segmentation_wrapper_factory)


def video_classification_wrapper_factory(dataset):
    if dataset.video_clips.output_format == "THWC":
        raise RuntimeError(
            f"{type(dataset).__name__} with `output_format='THWC'` is not supported by this wrapper, "
            f"since it is not compatible with the transformations. Please use `output_format='TCHW'` instead."
        )

    def wrapper(idx, sample):
        video, audio, label = sample

        video = datapoints.Video(video)

        return video, audio, label

    return wrapper


for dataset_cls in [
    datasets.HMDB51,
    datasets.Kinetics,
    datasets.UCF101,
]:
    WRAPPER_FACTORIES.register(dataset_cls)(video_classification_wrapper_factory)


@WRAPPER_FACTORIES.register(datasets.Caltech101)
def caltech101_wrapper_factory(dataset):
    if "annotation" in dataset.target_type:
        raise_not_supported("Caltech101 dataset with `target_type=['annotation', ...]`")

    return classification_wrapper_factory(dataset)


@WRAPPER_FACTORIES.register(datasets.CocoDetection)
def coco_dectection_wrapper_factory(dataset):
    def segmentation_to_mask(segmentation, *, spatial_size):
        from pycocotools import mask

        segmentation = (
            mask.frPyObjects(segmentation, *spatial_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *spatial_size))
        )
        return torch.from_numpy(mask.decode(segmentation))

    def wrapper(idx, sample):
        image_id = dataset.ids[idx]

        image, target = sample

        if not target:
            return image, dict(image_id=image_id)

        batched_target = list_of_dicts_to_dict_of_lists(target)

        batched_target["image_id"] = image_id

        spatial_size = tuple(F.get_spatial_size(image))
        batched_target["boxes"] = F.convert_format_bounding_box(
            datapoints.BoundingBox(
                batched_target["bbox"],
                format=datapoints.BoundingBoxFormat.XYWH,
                spatial_size=spatial_size,
            ),
            new_format=datapoints.BoundingBoxFormat.XYXY,
        )
        batched_target["masks"] = datapoints.Mask(
            torch.stack(
                [
                    segmentation_to_mask(segmentation, spatial_size=spatial_size)
                    for segmentation in batched_target["segmentation"]
                ]
            ),
        )
        batched_target["labels"] = torch.tensor(batched_target["category_id"])

        return image, batched_target

    return wrapper


WRAPPER_FACTORIES.register(datasets.CocoCaptions)(identity_wrapper_factory)


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


@WRAPPER_FACTORIES.register(datasets.VOCDetection)
def voc_detection_wrapper_factory(dataset):
    def wrapper(idx, sample):
        image, target = sample

        batched_instances = list_of_dicts_to_dict_of_lists(target["annotation"]["object"])

        target["boxes"] = datapoints.BoundingBox(
            [
                [int(bndbox[part]) for part in ("xmin", "ymin", "xmax", "ymax")]
                for bndbox in batched_instances["bndbox"]
            ],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(image.height, image.width),
        )
        target["labels"] = torch.tensor(
            [VOC_DETECTION_CATEGORY_TO_IDX[category] for category in batched_instances["name"]]
        )

        return image, target

    return wrapper


@WRAPPER_FACTORIES.register(datasets.SBDataset)
def sbd_wrapper(dataset):
    if dataset.mode == "boundaries":
        raise_not_supported("SBDataset with mode='boundaries'")

    return segmentation_wrapper_factory(dataset)


@WRAPPER_FACTORIES.register(datasets.CelebA)
def celeba_wrapper_factory(dataset):
    if any(target_type in dataset.target_type for target_type in ["attr", "landmarks"]):
        raise_not_supported("`CelebA` dataset with `target_type=['attr', 'landmarks', ...]`")

    def wrapper(idx, sample):
        image, target = sample

        target = wrap_target_by_type(
            target,
            target_types=dataset.target_type,
            type_wrappers={
                "bbox": lambda item: F.convert_format_bounding_box(
                    datapoints.BoundingBox(
                        item,
                        format=datapoints.BoundingBoxFormat.XYWH,
                        spatial_size=(image.height, image.width),
                    ),
                    new_format=datapoints.BoundingBoxFormat.XYXY,
                ),
            },
        )

        return image, target

    return wrapper


KITTI_CATEGORIES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
KITTI_CATEGORY_TO_IDX = dict(zip(KITTI_CATEGORIES, range(len(KITTI_CATEGORIES))))


@WRAPPER_FACTORIES.register(datasets.Kitti)
def kitti_wrapper_factory(dataset):
    def wrapper(idx, sample):
        image, target = sample

        if target is not None:
            target = list_of_dicts_to_dict_of_lists(target)

            target["boxes"] = datapoints.BoundingBox(
                target["bbox"], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(image.height, image.width)
            )
            target["labels"] = torch.tensor([KITTI_CATEGORY_TO_IDX[category] for category in target["type"]])

        return image, target

    return wrapper


@WRAPPER_FACTORIES.register(datasets.OxfordIIITPet)
def oxford_iiit_pet_wrapper_factor(dataset):
    def wrapper(idx, sample):
        image, target = sample

        if target is not None:
            target = wrap_target_by_type(
                target,
                target_types=dataset._target_types,
                type_wrappers={
                    "segmentation": pil_image_to_mask,
                },
            )

        return image, target

    return wrapper


@WRAPPER_FACTORIES.register(datasets.Cityscapes)
def cityscapes_wrapper_factory(dataset):
    if any(target_type in dataset.target_type for target_type in ["polygon", "color"]):
        raise_not_supported("`Cityscapes` dataset with `target_type=['polygon', 'color', ...]`")

    def instance_segmentation_wrapper(mask):
        # See https://github.com/mcordts/cityscapesScripts/blob/8da5dd00c9069058ccc134654116aac52d4f6fa2/cityscapesscripts/preparation/json2instanceImg.py#L7-L21
        data = pil_image_to_mask(mask)
        masks = []
        labels = []
        for id in data.unique():
            masks.append(data == id)
            label = id
            if label >= 1_000:
                label //= 1_000
            labels.append(label)
        return dict(masks=datapoints.Mask(torch.stack(masks)), labels=torch.stack(labels))

    def wrapper(idx, sample):
        image, target = sample

        target = wrap_target_by_type(
            target,
            target_types=dataset.target_type,
            type_wrappers={
                "instance": instance_segmentation_wrapper,
                "semantic": pil_image_to_mask,
            },
        )

        return image, target

    return wrapper


@WRAPPER_FACTORIES.register(datasets.WIDERFace)
def widerface_wrapper(dataset):
    def wrapper(idx, sample):
        image, target = sample

        if target is not None:
            target["bbox"] = F.convert_format_bounding_box(
                datapoints.BoundingBox(
                    target["bbox"], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=(image.height, image.width)
                ),
                new_format=datapoints.BoundingBoxFormat.XYXY,
            )

        return image, target

    return wrapper
