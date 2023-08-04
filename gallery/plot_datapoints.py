"""
==============
Datapoints FAQ
==============

Datapoints are Tensor subclasses introduced together with
``torchvision.transforms.v2``. This example showcases what these datapoints are
and how they behave.

.. warning::

    **Intended Audience** Unless you're writing your own transforms or your own datapoints, you
    probably do not need to read this guide. This is a fairly low-level topic
    that most users will not need to worry about: you do not need to understand
    the internals of datapoints to efficiently rely on
    ``torchvision.transforms.v2``. It may however be useful for advanced users
    trying to implement their own datasets, transforms, or work directly with
    the datapoints.
"""

# %%
import PIL.Image

import torch
import torchvision

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F


# %%
# What are datapoints?
# --------------------
#
# Datapoints are zero-copy tensor subclasses:

tensor = torch.rand(3, 256, 256)
image = datapoints.Image(tensor)

assert isinstance(image, torch.Tensor)
assert image.data_ptr() == tensor.data_ptr()

# %%
# Under the hood, they are needed in :mod:`torchvision.transforms.v2` to correctly dispatch to the appropriate function
# for the input data.
#
# What can I do with a datapoint?
# -------------------------------
#
# Datapoints look and feel just like regular tensors - they **are** tensors.
# Everything that is supported on a plain :class:`torch.Tensor` like ``.sum()`` or
# any ``torch.*`` operator will also works on datapoints. See
# :ref:`datapoint_unwrapping_behaviour` for a few gotchas.

# %%
#
# What datapoints are supported?
# ------------------------------
#
# So far :mod:`torchvision.datapoints` supports four types of datapoints:
#
# * :class:`~torchvision.datapoints.Image`
# * :class:`~torchvision.datapoints.Video`
# * :class:`~torchvision.datapoints.BoundingBoxes`
# * :class:`~torchvision.datapoints.Mask`
#
# .. _datapoint_creation:
#
# How do I construct a datapoint?
# -------------------------------
#
# Using the constructor
# ^^^^^^^^^^^^^^^^^^^^^
#
# Each datapoint class takes any tensor-like data that can be turned into a :class:`~torch.Tensor`

image = datapoints.Image([[[[0, 1], [1, 0]]]])
print(image)


# %%
# Similar to other PyTorch creations ops, the constructor also takes the ``dtype``, ``device``, and ``requires_grad``
# parameters.

float_image = datapoints.Image([[[0, 1], [1, 0]]], dtype=torch.float32, requires_grad=True)
print(float_image)


# %%
# In addition, :class:`~torchvision.datapoints.Image` and :class:`~torchvision.datapoints.Mask` can also take a
# :class:`PIL.Image.Image` directly:

image = datapoints.Image(PIL.Image.open("assets/astronaut.jpg"))
print(image.shape, image.dtype)

# %%
# Some datapoints require additional metadata to be passed in ordered to be constructed. For example,
# :class:`~torchvision.datapoints.BoundingBoxes` requires the coordinate format as well as the size of the
# corresponding image (``canvas_size``) alongside the actual values. These
# metadata are required to properly transform the bounding boxes.

bboxes = datapoints.BoundingBoxes(
    [[17, 16, 344, 495], [0, 10, 0, 10]],
    format=datapoints.BoundingBoxFormat.XYXY,
    canvas_size=image.shape[-2:]
)
print(bboxes)

# %%
# Using the ``wrap_like()`` class method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can also use the ``wrap_like()`` class method to wrap a tensor object
# into a datapoint. This is useful when you already have an object of the
# desired type, which typically happens when writing transforms: you just want
# to wrap the output like the input. This API is inspired by utils like
# :func:`torch.zeros_like`:

new_bboxes = torch.tensor([0, 20, 30, 40])
new_bboxes = datapoints.BoundingBoxes.wrap_like(bboxes, new_bboxes)
assert isinstance(new_bboxes, datapoints.BoundingBoxes)
assert new_bboxes.canvas_size == bboxes.canvas_size


# %%
# The metadata of ``new_bboxes`` is the same as ``bboxes``, but you could pass
# it as a parameter to override it. Check the
# :meth:`~torchvision.datapoints.BoundingBoxes.wrap_like` documentation for
# more details.
#
# Do I have to wrap the output of the datasets myself?
# ----------------------------------------------------
#
# TODO: Move this in another guide - this is user-facing, not dev-facing.
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

        target["bboxes"] = datapoints.BoundingBoxes(
            bboxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            canvas_size=F.get_size(img),
        )
        target["labels"] = labels
        target["masks"] = datapoints.Mask(masks)

        ...

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        ...

# %%
# 2. Perform the wrapping inside a custom transformation at the beginning of your pipeline:


class WrapPennFudanDataset:
    def __call__(self, img, target):
        target["boxes"] = datapoints.BoundingBoxes(
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

# %%
# .. note::
#
#    If both :class:`~torchvision.datapoints.BoundingBoxes` and :class:`~torchvision.datapoints.Mask`'s are included in
#    the sample, ``torchvision.transforms.v2`` will transform them both. Meaning, if you don't need both, dropping or
#    at least not wrapping the obsolete parts, can lead to a significant performance boost.
#
#    For example, if you are using the ``PennFudanDataset`` for object detection, not wrapping the masks avoids
#    transforming them over and over again in the pipeline just to ultimately ignoring them. In general, it would be
#    even better to not load the masks at all, but this is not possible in this example, since the bounding boxes are
#    generated from the masks.
#
# .. _datapoint_unwrapping_behaviour:
#
# I had a Datapoint but now I have a Tensor. Help!
# ------------------------------------------------
#
# For a lot of operations involving datapoints, we cannot safely infer whether
# the result should retain the datapoint type, so we choose to return a plain
# tensor instead of a datapoint (this might change, see note below):


assert isinstance(bboxes, datapoints.BoundingBoxes)

# Shift bboxes by 3 pixels in both H and W
new_bboxes = bboxes + 3

assert isinstance(new_bboxes, torch.Tensor) and not isinstance(new_bboxes, datapoints.BoundingBoxes)

# %%
# If you're writing your own custom transforms or code involving datapoints, you
# can re-wrap the output into a datapoint by just calling their constructor, or
# by using the ``.wrap_like()`` class method:

new_bboxes = bboxes + 3
new_bboxes = datapoints.BoundingBoxes.wrap_like(bboxes, new_bboxes)
assert isinstance(new_bboxes, datapoints.BoundingBoxes)

# %%
# See more details above in :ref:`datapoint_creation`.
#
# .. note::
#
#    You never need to re-wrap manually if you're using the built-in transforms
#    or their functional equivalents: this is automatically taken care of for
#    you.
#
# .. note::
#
#    This "unwrapping" behaviour is something we're actively seeking feedback on. If you find this surprising or if you
#    have any suggestions on how to better support your use-cases, please reach out to us via this issue:
#    https://github.com/pytorch/vision/issues/7319
#
# There are a few exceptions to this "unwrapping" rule:
#
# 1. Operations like :meth:`~torch.Tensor.clone`, :meth:`~torch.Tensor.to`,
#    :meth:`torch.Tensor.detach` and :meth:`~torch.Tensor.requires_grad_` retain
#    the datapoint type.
# 2. Inplace operations on datapoints like ``.add_()`` preserve they type. However,
#    the **returned** value of inplace operations will be unwrapped into a pure
#    tensor:

image = datapoints.Image([[[0, 1], [1, 0]]])

new_image = image.add_(1).mul_(2)

# image got transformed in-place and is still an Image datapoint, but new_image
# is a Tensor. They share the same underlying data and they're equal, just
# different classes.
assert isinstance(image, datapoints.Image)
print(image)

assert isinstance(new_image, torch.Tensor) and not isinstance(new_image, datapoints.Image)
assert (new_image == image).all()
assert new_image.data_ptr() == image.data_ptr()
