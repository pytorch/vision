"""
=============
TVTensors FAQ
=============

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_tv_tensors.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_tv_tensors.py>` to download the full example code.


TVTensors are Tensor subclasses introduced together with
``torchvision.transforms.v2``. This example showcases what these TVTensors are
and how they behave.

.. warning::

    **Intended Audience** Unless you're writing your own transforms or your own TVTensors, you
    probably do not need to read this guide. This is a fairly low-level topic
    that most users will not need to worry about: you do not need to understand
    the internals of TVTensors to efficiently rely on
    ``torchvision.transforms.v2``. It may however be useful for advanced users
    trying to implement their own datasets, transforms, or work directly with
    the TVTensors.
"""

# %%
import PIL.Image

import torch
from torchvision import tv_tensors


# %%
# What are TVTensors?
# -------------------
#
# TVTensors are zero-copy tensor subclasses:

tensor = torch.rand(3, 256, 256)
image = tv_tensors.Image(tensor)

assert isinstance(image, torch.Tensor)
assert image.data_ptr() == tensor.data_ptr()

# %%
# Under the hood, they are needed in :mod:`torchvision.transforms.v2` to correctly dispatch to the appropriate function
# for the input data.
#
# :mod:`torchvision.tv_tensors` supports four types of TVTensors:
#
# * :class:`~torchvision.tv_tensors.Image`
# * :class:`~torchvision.tv_tensors.Video`
# * :class:`~torchvision.tv_tensors.BoundingBoxes`
# * :class:`~torchvision.tv_tensors.Mask`
#
# What can I do with a TVTensor?
# ------------------------------
#
# TVTensors look and feel just like regular tensors - they **are** tensors.
# Everything that is supported on a plain :class:`torch.Tensor` like ``.sum()`` or
# any ``torch.*`` operator will also work on TVTensors. See
# :ref:`tv_tensor_unwrapping_behaviour` for a few gotchas.

# %%
# .. _tv_tensor_creation:
#
# How do I construct a TVTensor?
# ------------------------------
#
# Using the constructor
# ^^^^^^^^^^^^^^^^^^^^^
#
# Each TVTensor class takes any tensor-like data that can be turned into a :class:`~torch.Tensor`

image = tv_tensors.Image([[[[0, 1], [1, 0]]]])
print(image)


# %%
# Similar to other PyTorch creations ops, the constructor also takes the ``dtype``, ``device``, and ``requires_grad``
# parameters.

float_image = tv_tensors.Image([[[0, 1], [1, 0]]], dtype=torch.float32, requires_grad=True)
print(float_image)


# %%
# In addition, :class:`~torchvision.tv_tensors.Image` and :class:`~torchvision.tv_tensors.Mask` can also take a
# :class:`PIL.Image.Image` directly:

image = tv_tensors.Image(PIL.Image.open("../assets/astronaut.jpg"))
print(image.shape, image.dtype)

# %%
# Some TVTensors require additional metadata to be passed in ordered to be constructed. For example,
# :class:`~torchvision.tv_tensors.BoundingBoxes` requires the coordinate format as well as the size of the
# corresponding image (``canvas_size``) alongside the actual values. These
# metadata are required to properly transform the bounding boxes.

bboxes = tv_tensors.BoundingBoxes(
    [[17, 16, 344, 495], [0, 10, 0, 10]],
    format=tv_tensors.BoundingBoxFormat.XYXY,
    canvas_size=image.shape[-2:]
)
print(bboxes)

# %%
# Using ``tv_tensors.wrap()``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can also use the :func:`~torchvision.tv_tensors.wrap` function to wrap a tensor object
# into a TVTensor. This is useful when you already have an object of the
# desired type, which typically happens when writing transforms: you just want
# to wrap the output like the input.

new_bboxes = torch.tensor([0, 20, 30, 40])
new_bboxes = tv_tensors.wrap(new_bboxes, like=bboxes)
assert isinstance(new_bboxes, tv_tensors.BoundingBoxes)
assert new_bboxes.canvas_size == bboxes.canvas_size

# %%
# The metadata of ``new_bboxes`` is the same as ``bboxes``, but you could pass
# it as a parameter to override it.
#
# .. _tv_tensor_unwrapping_behaviour:
#
# I had a TVTensor but now I have a Tensor. Help!
# -----------------------------------------------
#
# By default, operations on :class:`~torchvision.tv_tensors.TVTensor` objects
# will return a pure Tensor:


assert isinstance(bboxes, tv_tensors.BoundingBoxes)

# Shift bboxes by 3 pixels in both H and W
new_bboxes = bboxes + 3

assert isinstance(new_bboxes, torch.Tensor)
assert not isinstance(new_bboxes, tv_tensors.BoundingBoxes)

# %%
# .. note::
#
#    This behavior only affects native ``torch`` operations. If you are using
#    the built-in ``torchvision`` transforms or functionals, you will always get
#    as output the same type that you passed as input (pure ``Tensor`` or
#    ``TVTensor``).

# %%
# But I want a TVTensor back!
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can re-wrap a pure tensor into a TVTensor by just calling the TVTensor
# constructor, or by using the :func:`~torchvision.tv_tensors.wrap` function
# (see more details above in :ref:`tv_tensor_creation`):

new_bboxes = bboxes + 3
new_bboxes = tv_tensors.wrap(new_bboxes, like=bboxes)
assert isinstance(new_bboxes, tv_tensors.BoundingBoxes)

# %%
# Alternatively, you can use the :func:`~torchvision.tv_tensors.set_return_type`
# as a global config setting for the whole program, or as a context manager
# (read its docs to learn more about caveats):

with tv_tensors.set_return_type("TVTensor"):
    new_bboxes = bboxes + 3
assert isinstance(new_bboxes, tv_tensors.BoundingBoxes)

# %%
# Why is this happening?
# ^^^^^^^^^^^^^^^^^^^^^^
#
# **For performance reasons**. :class:`~torchvision.tv_tensors.TVTensor`
# classes are Tensor subclasses, so any operation involving a
# :class:`~torchvision.tv_tensors.TVTensor` object will go through the
# `__torch_function__
# <https://pytorch.org/docs/stable/notes/extending.html#extending-torch>`_
# protocol. This induces a small overhead, which we want to avoid when possible.
# This doesn't matter for built-in ``torchvision`` transforms because we can
# avoid the overhead there, but it could be a problem in your model's
# ``forward``.
#
# **The alternative isn't much better anyway.** For every operation where
# preserving the :class:`~torchvision.tv_tensors.TVTensor` type makes
# sense, there are just as many operations where returning a pure Tensor is
# preferable: for example, is ``img.sum()`` still an :class:`~torchvision.tv_tensors.Image`?
# If we were to preserve :class:`~torchvision.tv_tensors.TVTensor` types all
# the way, even model's logits or the output of the loss function would end up
# being of type :class:`~torchvision.tv_tensors.Image`, and surely that's not
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
# the TVTensor type.
#
# Inplace operations on TVTensors like ``obj.add_()`` will preserve the type of
# ``obj``. However, the **returned** value of inplace operations will be a pure
# tensor:

image = tv_tensors.Image([[[0, 1], [1, 0]]])

new_image = image.add_(1).mul_(2)

# image got transformed in-place and is still a TVTensor Image, but new_image
# is a Tensor. They share the same underlying data and they're equal, just
# different classes.
assert isinstance(image, tv_tensors.Image)
print(image)

assert isinstance(new_image, torch.Tensor) and not isinstance(new_image, tv_tensors.Image)
assert (new_image == image).all()
assert new_image.data_ptr() == image.data_ptr()
