"""
=================
VisionTensors FAQ
=================

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_vision_tensors.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_vision_tensors.py>` to download the full example code.


VisionTensors are Tensor subclasses introduced together with
``torchvision.transforms.v2``. This example showcases what these vision_tensors are
and how they behave.

.. warning::

    **Intended Audience** Unless you're writing your own transforms or your own vision_tensors, you
    probably do not need to read this guide. This is a fairly low-level topic
    that most users will not need to worry about: you do not need to understand
    the internals of vision_tensors to efficiently rely on
    ``torchvision.transforms.v2``. It may however be useful for advanced users
    trying to implement their own datasets, transforms, or work directly with
    the vision_tensors.
"""

# %%
import PIL.Image

import torch
from torchvision import vision_tensors


# %%
# What are vision_tensors?
# ------------------------
#
# VisionTensors are zero-copy tensor subclasses:

tensor = torch.rand(3, 256, 256)
image = vision_tensors.Image(tensor)

assert isinstance(image, torch.Tensor)
assert image.data_ptr() == tensor.data_ptr()

# %%
# Under the hood, they are needed in :mod:`torchvision.transforms.v2` to correctly dispatch to the appropriate function
# for the input data.
#
# :mod:`torchvision.vision_tensors` supports four types of vision_tensors:
#
# * :class:`~torchvision.vision_tensors.Image`
# * :class:`~torchvision.vision_tensors.Video`
# * :class:`~torchvision.vision_tensors.BoundingBoxes`
# * :class:`~torchvision.vision_tensors.Mask`
#
# What can I do with a vision_tensor?
# -----------------------------------
#
# VisionTensors look and feel just like regular tensors - they **are** tensors.
# Everything that is supported on a plain :class:`torch.Tensor` like ``.sum()`` or
# any ``torch.*`` operator will also work on vision_tensors. See
# :ref:`vision_tensor_unwrapping_behaviour` for a few gotchas.

# %%
# .. _vision_tensor_creation:
#
# How do I construct a vision_tensor?
# -----------------------------------
#
# Using the constructor
# ^^^^^^^^^^^^^^^^^^^^^
#
# Each vision_tensor class takes any tensor-like data that can be turned into a :class:`~torch.Tensor`

image = vision_tensors.Image([[[[0, 1], [1, 0]]]])
print(image)


# %%
# Similar to other PyTorch creations ops, the constructor also takes the ``dtype``, ``device``, and ``requires_grad``
# parameters.

float_image = vision_tensors.Image([[[0, 1], [1, 0]]], dtype=torch.float32, requires_grad=True)
print(float_image)


# %%
# In addition, :class:`~torchvision.vision_tensors.Image` and :class:`~torchvision.vision_tensors.Mask` can also take a
# :class:`PIL.Image.Image` directly:

image = vision_tensors.Image(PIL.Image.open("../assets/astronaut.jpg"))
print(image.shape, image.dtype)

# %%
# Some vision_tensors require additional metadata to be passed in ordered to be constructed. For example,
# :class:`~torchvision.vision_tensors.BoundingBoxes` requires the coordinate format as well as the size of the
# corresponding image (``canvas_size``) alongside the actual values. These
# metadata are required to properly transform the bounding boxes.

bboxes = vision_tensors.BoundingBoxes(
    [[17, 16, 344, 495], [0, 10, 0, 10]],
    format=vision_tensors.BoundingBoxFormat.XYXY,
    canvas_size=image.shape[-2:]
)
print(bboxes)

# %%
# Using ``vision_tensors.wrap()``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can also use the :func:`~torchvision.vision_tensors.wrap` function to wrap a tensor object
# into a vision_tensor. This is useful when you already have an object of the
# desired type, which typically happens when writing transforms: you just want
# to wrap the output like the input.

new_bboxes = torch.tensor([0, 20, 30, 40])
new_bboxes = vision_tensors.wrap(new_bboxes, like=bboxes)
assert isinstance(new_bboxes, vision_tensors.BoundingBoxes)
assert new_bboxes.canvas_size == bboxes.canvas_size

# %%
# The metadata of ``new_bboxes`` is the same as ``bboxes``, but you could pass
# it as a parameter to override it.
#
# .. _vision_tensor_unwrapping_behaviour:
#
# I had a VisionTensor but now I have a Tensor. Help!
# ---------------------------------------------------
#
# By default, operations on :class:`~torchvision.vision_tensors.VisionTensor` objects
# will return a pure Tensor:


assert isinstance(bboxes, vision_tensors.BoundingBoxes)

# Shift bboxes by 3 pixels in both H and W
new_bboxes = bboxes + 3

assert isinstance(new_bboxes, torch.Tensor)
assert not isinstance(new_bboxes, vision_tensors.BoundingBoxes)

# %%
# .. note::
#
#    This behavior only affects native ``torch`` operations. If you are using
#    the built-in ``torchvision`` transforms or functionals, you will always get
#    as output the same type that you passed as input (pure ``Tensor`` or
#    ``VisionTensor``).

# %%
# But I want a VisionTensor back!
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can re-wrap a pure tensor into a vision_tensor by just calling the vision_tensor
# constructor, or by using the :func:`~torchvision.vision_tensors.wrap` function
# (see more details above in :ref:`vision_tensor_creation`):

new_bboxes = bboxes + 3
new_bboxes = vision_tensors.wrap(new_bboxes, like=bboxes)
assert isinstance(new_bboxes, vision_tensors.BoundingBoxes)

# %%
# Alternatively, you can use the :func:`~torchvision.vision_tensors.set_return_type`
# as a global config setting for the whole program, or as a context manager
# (read its docs to learn more about caveats):

with vision_tensors.set_return_type("vision_tensor"):
    new_bboxes = bboxes + 3
assert isinstance(new_bboxes, vision_tensors.BoundingBoxes)

# %%
# Why is this happening?
# ^^^^^^^^^^^^^^^^^^^^^^
#
# **For performance reasons**. :class:`~torchvision.vision_tensors.VisionTensor`
# classes are Tensor subclasses, so any operation involving a
# :class:`~torchvision.vision_tensors.VisionTensor` object will go through the
# `__torch_function__
# <https://pytorch.org/docs/stable/notes/extending.html#extending-torch>`_
# protocol. This induces a small overhead, which we want to avoid when possible.
# This doesn't matter for built-in ``torchvision`` transforms because we can
# avoid the overhead there, but it could be a problem in your model's
# ``forward``.
#
# **The alternative isn't much better anyway.** For every operation where
# preserving the :class:`~torchvision.vision_tensors.VisionTensor` type makes
# sense, there are just as many operations where returning a pure Tensor is
# preferable: for example, is ``img.sum()`` still an :class:`~torchvision.vision_tensors.Image`?
# If we were to preserve :class:`~torchvision.vision_tensors.VisionTensor` types all
# the way, even model's logits or the output of the loss function would end up
# being of type :class:`~torchvision.vision_tensors.Image`, and surely that's not
# desirable.
#
# .. note::
#
#    This behaviour is something we're actively seeking feedback on. If you find this surprising or if you
#    have any suggestions on how to better support your use-cases, please reach out to us via this issue:
#    https://github.com/pytorch/vision/issues/7319
#
# Exceptions
# ^^^^^^^^^^
#
# There are a few exceptions to this "unwrapping" rule:
# :meth:`~torch.Tensor.clone`, :meth:`~torch.Tensor.to`,
# :meth:`torch.Tensor.detach`, and :meth:`~torch.Tensor.requires_grad_` retain
# the vision_tensor type.
#
# Inplace operations on vision_tensors like ``obj.add_()`` will preserve the type of
# ``obj``. However, the **returned** value of inplace operations will be a pure
# tensor:

image = vision_tensors.Image([[[0, 1], [1, 0]]])

new_image = image.add_(1).mul_(2)

# image got transformed in-place and is still an Image vision_tensor, but new_image
# is a Tensor. They share the same underlying data and they're equal, just
# different classes.
assert isinstance(image, vision_tensors.Image)
print(image)

assert isinstance(new_image, torch.Tensor) and not isinstance(new_image, vision_tensors.Image)
assert (new_image == image).all()
assert new_image.data_ptr() == image.data_ptr()
