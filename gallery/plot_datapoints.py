"""
==============
Datapoints FAQ
==============

The :mod:`torchvision.datapoints` namespace was introduced together with ``torchvision.transforms.v2``. This example
showcases what these datapoints are and how they behave. This is a fairly low-level topic that most users will not need
to worry about: you do not need to understand the internals of datapoints to efficiently rely on
``torchvision.transforms.v2``. It may however be useful for advanced users trying to implement their own datasets,
transforms, or work directly with the datapoints.
"""

import PIL.Image

import torch
import torchvision

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F


########################################################################################################################
# What are datapoints?
# --------------------
#
# Datapoints are zero-copy tensor subclasses:

tensor = torch.rand(3, 256, 256)
image = datapoints.Image(tensor)

assert isinstance(image, torch.Tensor)
assert image.data_ptr() == tensor.data_ptr()


########################################################################################################################
# Under the hood, they are needed in :mod:`torchvision.transforms.v2` to correctly dispatch to the appropriate function
# for the input data.
#
# What datapoints are supported?
# ------------------------------
#
# So far :mod:`torchvision.datapoints` supports four types of datapoints:
#
# * :class:`~torchvision.datapoints.Image`
# * :class:`~torchvision.datapoints.Video`
# * :class:`~torchvision.datapoints.BoundingBox`
# * :class:`~torchvision.datapoints.Mask`
#
# How do I construct a datapoint?
# -------------------------------
#
# Each datapoint class takes any tensor-like data that can be turned into a :class:`~torch.Tensor`

image = datapoints.Image([[[[0, 1], [1, 0]]]])
print(image)


########################################################################################################################
# Similar to other PyTorch creations ops, the constructor also takes the ``dtype``, ``device``, and ``requires_grad``
# parameters.

float_image = datapoints.Image([[[0, 1], [1, 0]]], dtype=torch.float32, requires_grad=True)
print(float_image)


########################################################################################################################
# In addition, :class:`~torchvision.datapoints.Image` and :class:`~torchvision.datapoints.Mask` also take a
# :class:`PIL.Image.Image` directly:

image = datapoints.Image(PIL.Image.open("assets/astronaut.jpg"))
print(image.shape, image.dtype)

########################################################################################################################
# In general, the datapoints can also store additional metadata that complements the underlying tensor. For example,
# :class:`~torchvision.datapoints.BoundingBox` stores the coordinate format as well as the canvas size of the
# corresponding image alongside the actual values:

bounding_box = datapoints.BoundingBox(
    [17, 16, 344, 495], format=datapoints.BoundingBoxFormat.XYXY, canvas_size=F.get_size(image)
)
print(bounding_box)


########################################################################################################################
# Do I have to wrap the output of the datasets myself?
# ----------------------------------------------------
#
# Only if you are using custom datasets. For the built-in ones, you can use
# :func:`torchvision.datasets.wrap_dataset_for_transforms_v2`. Note that the function also supports subclasses of the
# built-in datasets. Meaning, if your custom dataset subclasses from a built-in one and the output type is the same, you
# also don't have to wrap manually.
#
# If you have a custom dataset, for example the ``PennFudanDataset`` from
# `this tutorial <https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html>`_, you have two options:
#
# 1. Perform the wrapping inside ``__getitem__``:

class PennFudanDataset(torch.utils.data.Dataset):
    ...

    def __getitem__(self, item):
        ...

        target["boxes"] = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            canvas_size=F.get_size(img),
        )
        target["labels"] = labels
        target["masks"] = datapoints.Mask(masks)

        ...

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        ...

########################################################################################################################
# 2. Perform the wrapping inside a custom transformation at the beginning of your pipeline:


class WrapPennFudanDataset:
    def __call__(self, img, target):
        target["boxes"] = datapoints.BoundingBox(
            target["boxes"],
            format=datapoints.BoundingBoxFormat.XYXY,
            canvas_size=F.get_size(img),
        )
        target["masks"] = datapoints.Mask(target["masks"])
        return img, target


...


def get_transform(train):
    transforms = []
    transforms.append(WrapPennFudanDataset())
    transforms.append(T.PILToTensor())
    ...

########################################################################################################################
# .. note::
#
#    If both :class:`~torchvision.datapoints.BoundingBox`'es and :class:`~torchvision.datapoints.Mask`'s are included in
#    the sample, ``torchvision.transforms.v2`` will transform them both. Meaning, if you don't need both, dropping or
#    at least not wrapping the obsolete parts, can lead to a significant performance boost.
#
#    For example, if you are using the ``PennFudanDataset`` for object detection, not wrapping the masks avoids
#    transforming them over and over again in the pipeline just to ultimately ignoring them. In general, it would be
#    even better to not load the masks at all, but this is not possible in this example, since the bounding boxes are
#    generated from the masks.
#
# How do the datapoints behave inside a computation?
# --------------------------------------------------
#
# Datapoints look and feel just like regular tensors. Everything that is supported on a plain :class:`torch.Tensor`
# also works on datapoints.
# Since for most operations involving datapoints, it cannot be safely inferred whether the result should retain the
# datapoint type, we choose to return a plain tensor instead of a datapoint (this might change, see note below):


assert isinstance(image, datapoints.Image)

new_image = image + 0

assert isinstance(new_image, torch.Tensor) and not isinstance(new_image, datapoints.Image)

########################################################################################################################
# .. note::
#
#    This "unwrapping" behaviour is something we're actively seeking feedback on. If you find this surprising or if you
#    have any suggestions on how to better support your use-cases, please reach out to us via this issue:
#    https://github.com/pytorch/vision/issues/7319
#
# There are two exceptions to this rule:
#
# 1. The operations :meth:`~torch.Tensor.clone`, :meth:`~torch.Tensor.to`, and :meth:`~torch.Tensor.requires_grad_`
#    retain the datapoint type.
# 2. Inplace operations on datapoints cannot change the type of the datapoint they are called on. However, if you use
#    the flow style, the returned value will be unwrapped:

image = datapoints.Image([[[0, 1], [1, 0]]])

new_image = image.add_(1).mul_(2)

assert isinstance(image, torch.Tensor)
print(image)

assert isinstance(new_image, torch.Tensor) and not isinstance(new_image, datapoints.Image)
assert (new_image == image).all()
