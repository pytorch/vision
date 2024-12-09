"""
===================================
How to write your own v2 transforms
===================================

.. note::
    Try on `Colab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_custom_transforms.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_custom_transforms.py>` to download the full example code.

This guide explains how to write transforms that are compatible with the
torchvision transforms V2 API.
"""

# %%
from typing import Any, Dict, List

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


# %%
# Just create a ``nn.Module`` and override the ``forward`` method
# ===============================================================
#
# In most cases, this is all you're going to need, as long as you already know
# the structure of the input that your transform will expect. For example if
# you're just doing image classification, your transform will typically accept a
# single image as input, or a ``(img, label)`` input. So you can just hard-code
# your ``forward`` method to accept just that, e.g.
#
# .. code:: python
#
#     class MyCustomTransform(torch.nn.Module):
#         def forward(self, img, label):
#             # Do some transformations
#             return new_img, new_label
#
# .. note::
#
#     This means that if you have a custom transform that is already compatible
#     with the V1 transforms (those in ``torchvision.transforms``), it will
#     still work with the V2 transforms without any change!
#
# We will illustrate this more completely below with a typical detection case,
# where our samples are just images, bounding boxes and labels:

class MyCustomTransform(torch.nn.Module):
    def forward(self, img, bboxes, label):  # we assume inputs are always structured like this
        print(
            f"I'm transforming an image of shape {img.shape} "
            f"with bboxes = {bboxes}\n{label = }"
        )
        # Do some transformations. Here, we're just passing though the input
        return img, bboxes, label


transforms = v2.Compose([
    MyCustomTransform(),
    v2.RandomResizedCrop((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=1),
    v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

H, W = 256, 256
img = torch.rand(3, H, W)
bboxes = tv_tensors.BoundingBoxes(
    torch.tensor([[0, 10, 10, 20], [50, 50, 70, 70]]),
    format="XYXY",
    canvas_size=(H, W)
)
label = 3

out_img, out_bboxes, out_label = transforms(img, bboxes, label)
# %%
print(f"Output image shape: {out_img.shape}\nout_bboxes = {out_bboxes}\n{out_label = }")
# %%
# .. note::
#     While working with TVTensor classes in your code, make sure to
#     familiarize yourself with this section:
#     :ref:`tv_tensor_unwrapping_behaviour`
#
# Supporting arbitrary input structures
# =====================================
#
# In the section above, we have assumed that you already know the structure of
# your inputs and that you're OK with hard-coding this expected structure in
# your code. If you want your custom transforms to be as flexible as possible,
# this can be a bit limiting.
#
# A key feature of the builtin Torchvision V2 transforms is that they can accept
# arbitrary input structure and return the same structure as output (with
# transformed entries). For example, transforms can accept a single image, or a
# tuple of ``(img, label)``, or an arbitrary nested dictionary as input. Here's
# an example on the built-in transform :class:`~torchvision.transforms.v2.RandomHorizontalFlip`:

structured_input = {
    "img": img,
    "annotations": (bboxes, label),
    "something that will be ignored": (1, "hello"),
    "another tensor that is ignored": torch.arange(10),
}
structured_output = v2.RandomHorizontalFlip(p=1)(structured_input)

assert isinstance(structured_output, dict)
assert structured_output["something that will be ignored"] == (1, "hello")
assert (structured_output["another tensor that is ignored"] == torch.arange(10)).all()
print(f"The input bboxes are:\n{structured_input['annotations'][0]}")
print(f"The transformed bboxes are:\n{structured_output['annotations'][0]}")

# %%
# Basics: override the `transform()` method
# -----------------------------------------
#
# In order to support arbitrary inputs in your custom transform, you will need
# to inherit from :class:`~torchvision.transforms.v2.Transform` and override the
# `.transform()` method (not the `forward()` method!). Below is a basic example:


class MyCustomTransform(v2.Transform):
    def transform(self, inpt: Any, params: Dict[str, Any]):
        if type(inpt) == torch.Tensor:
            print(f"I'm transforming an image of shape {inpt.shape}")
            return inpt + 1  # dummy transformation
        elif isinstance(inpt, tv_tensors.BoundingBoxes):
            print(f"I'm transforming bounding boxes! {inpt.canvas_size = }")
            return tv_tensors.wrap(inpt + 100, like=inpt)  # dummy transformation


my_custom_transform = MyCustomTransform()
structured_output = my_custom_transform(structured_input)

assert isinstance(structured_output, dict)
assert structured_output["something that will be ignored"] == (1, "hello")
assert (structured_output["another tensor that is ignored"] == torch.arange(10)).all()
print(f"The input bboxes are:\n{structured_input['annotations'][0]}")
print(f"The transformed bboxes are:\n{structured_output['annotations'][0]}")

# %%
# An important thing to note is that when we call ``my_custom_transform`` on
# ``structured_input``, the input is flattened and then each individual part is
# passed to ``transform()``. That is, ``transform()``` receives the input image,
# then the bounding boxes, etc. Within ``transform()``, you can decide how to
# transform each input, based on their type.
#
# If you're curious why the other tensor (``torch.arange()``) didn't get passed
# to ``transform()``, see :ref:`this note <passthrough_heuristic>` for more
# details.
#
# Advanced: The ``make_params()`` method
# --------------------------------------
#
# The ``make_params()`` method is called internally before calling
# ``transform()`` on each input. This is typically useful to generate random
# parameter values. In the example below, we use it to randomly apply the
# transformation with a probability of 0.5


class MyRandomTransform(MyCustomTransform):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        apply_transform = (torch.rand(size=(1,)) < self.p).item()
        params = dict(apply_transform=apply_transform)
        return params

    def transform(self, inpt: Any, params: Dict[str, Any]):
        if not params["apply_transform"]:
            print("Not transforming anything!")
            return inpt
        else:
            return super().transform(inpt, params)


my_random_transform = MyRandomTransform()

torch.manual_seed(0)
_ = my_random_transform(structured_input)  # transforms
_ = my_random_transform(structured_input)  # doesn't transform

# %%
#
# .. note::
#
#     It's important for such random parameter generation to happen within
#     ``make_params()`` and not within ``transform()``, so that for a given
#     transform call, the same RNG applies to all the inputs in the same way. If
#     we were to perform the RNG within ``transform()``, we would risk e.g.
#     transforming the image while *not* transforming the bounding boxes.
#
# The ``make_params()`` method takes the list of all the inputs as parameter
# (each of the elements in this list will later be pased to ``transform()``).
# You can use ``flat_inputs`` to e.g. figure out the dimensions on the input,
# using :func:`~torchvision.transforms.v2.query_chw` or
# :func:`~torchvision.transforms.v2.query_size`.
#
# ``make_params()`` should return a dict (or actually, anything you want) that
# will then be passed to ``transform()``.
