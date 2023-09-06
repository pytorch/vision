"""
==================================
Getting started with transforms v2
==================================

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_transforms_v2.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_v2_transforms_plot_transforms_v2.py>` to download the full example code.

Most computer vision tasks are not supported out of the box by ``torchvision.transforms`` v1, since it only supports
images. ``torchvision.transforms.v2`` enables jointly transforming images, videos, bounding boxes, and masks. This
example showcases the core functionality of the new ``torchvision.transforms.v2`` API.
"""

import pathlib

import torch


def load_data():
    from torchvision.io import read_image
    from torchvision import datapoints
    from torchvision.ops import masks_to_boxes

    assets_directory = pathlib.Path("../assets")

    path = assets_directory / "FudanPed00054.png"
    image = datapoints.Image(read_image(str(path)))
    merged_masks = read_image(str(assets_directory / "FudanPed00054_mask.png"))

    labels = torch.unique(merged_masks)[1:]

    masks = datapoints.Mask(merged_masks == labels.view(-1, 1, 1))

    bounding_boxes = datapoints.BoundingBoxes(
        masks_to_boxes(masks), format=datapoints.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:]
    )

    return path, image, bounding_boxes, masks, labels


# %%
# The :mod:`torchvision.transforms.v2` API supports images, videos, bounding boxes, and instance and segmentation
# masks. Thus, it offers native support for many Computer Vision tasks, like image and video classification, object
# detection or instance and semantic segmentation. Still, the interface is the same, making
# :mod:`torchvision.transforms.v2` a drop-in replacement for the existing :mod:`torchvision.transforms` API, aka v1.

import torchvision.transforms.v2 as transforms

transform = transforms.Compose(
    [
        transforms.ColorJitter(contrast=0.5),
        transforms.RandomRotation(30),
        transforms.CenterCrop(480),
    ]
)

# %%
# :mod:`torchvision.transforms.v2` natively supports jointly transforming multiple inputs while making sure that
# potential random behavior is consistent across all inputs. However, it doesn't enforce a specific input structure or
# order.

path, image, bounding_boxes, masks, labels = load_data()

torch.manual_seed(0)
new_image = transform(image)  # Image Classification
new_image, new_bounding_boxes, new_labels = transform(image, bounding_boxes, labels)  # Object Detection
new_image, new_bounding_boxes, new_masks, new_labels = transform(
    image, bounding_boxes, masks, labels
)  # Instance Segmentation
new_image, new_target = transform((image, {"boxes": bounding_boxes, "labels": labels}))  # Arbitrary Structure

# %%
# Under the hood, :mod:`torchvision.transforms.v2` relies on :mod:`torchvision.datapoints` for the dispatch to the
# appropriate function for the input data: :ref:`sphx_glr_auto_examples_v2_transforms_plot_datapoints.py`. Note however, that as
# regular user, you likely don't have to touch this yourself. See
# :ref:`sphx_glr_auto_examples_v2_transforms_plot_transforms_v2_e2e.py`.
#
# All "foreign" types like :class:`str`'s or :class:`pathlib.Path`'s are passed through, allowing to store extra
# information directly with the sample:

sample = {"path": path, "image": image}
new_sample = transform(sample)

assert new_sample["path"] is sample["path"]

# %%
# As stated above, :mod:`torchvision.transforms.v2` is a drop-in replacement for :mod:`torchvision.transforms` and thus
# also supports transforming plain :class:`torch.Tensor`'s as image or video if applicable. This is achieved with a
# simple heuristic:
#
# * If we find an explicit image or video (:class:`torchvision.datapoints.Image`, :class:`torchvision.datapoints.Video`,
#   or :class:`PIL.Image.Image`) in the input, all other plain tensors are passed through.
# * If there is no explicit image or video, only the first plain :class:`torch.Tensor` will be transformed as image or
#   video, while all others will be passed through.

plain_tensor_image = torch.rand(image.shape)

print(image.shape, plain_tensor_image.shape)

# passing a plain tensor together with an explicit image, will not transform the former
plain_tensor_image, image = transform(plain_tensor_image, image)

print(image.shape, plain_tensor_image.shape)

# passing a plain tensor without an explicit image, will transform the former
plain_tensor_image, _ = transform(plain_tensor_image, bounding_boxes)

print(image.shape, plain_tensor_image.shape)
