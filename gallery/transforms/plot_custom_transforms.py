"""
===================================
How to write your own v2 transforms
===================================

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_custom_transforms.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_custom_transforms.py>` to download the full example code.

This guide explains how to write transforms that are compatible with the
torchvision transforms V2 API.
"""

# %%
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
# tuple of ``(img, label)``, or an arbitrary nested dictionary as input:

structured_input = {
    "img": img,
    "annotations": (bboxes, label),
    "something_that_will_be_ignored": (1, "hello")
}
structured_output = v2.RandomHorizontalFlip(p=1)(structured_input)

assert isinstance(structured_output, dict)
assert structured_output["something_that_will_be_ignored"] == (1, "hello")
print(f"The transformed bboxes are:\n{structured_output['annotations'][0]}")

# %%
# If you want to reproduce this behavior in your own transform, we invite you to
# look at our `code
# <https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py>`_
# and adapt it to your needs.
#
# In brief, the core logic is to unpack the input into a flat list using `pytree
# <https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py>`_, and
# then transform only the entries that can be transformed (the decision is made
# based on the **class** of the entries, as all TVTensors are
# tensor-subclasses) plus some custom logic that is out of score here - check the
# code for details. The (potentially transformed) entries are then repacked and
# returned, in the same structure as the input.
#
# We do not provide public dev-facing tools to achieve that at this time, but if
# this is something that would be valuable to you, please let us know by opening
# an issue on our `GitHub repo <https://github.com/pytorch/vision/issues>`_.
