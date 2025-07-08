"""
===============================================================
Transforms on Rotated Bounding Boxes
===============================================================

This example illustrates how to define and use rotated bounding boxes. We'll
cover how to define them, demonstrate their usage with some of the existing
transforms, and finally some of their unique behavior in comparision to
standard bounding boxes.

First, a bit of setup code:
"""

# %%
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from helpers import plot

plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["savefig.bbox"] = "tight"

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
orig_img = Image.open(Path('../assets') / 'leaning_tower.jpg')

# %%
# Creating a Rotated Bounding Box
# -------------------------------
# Rotated bounding boxes are created by instantiating the
# :class:`~torchvision.tv_tensors.BoundingBoxes` class. It's the `format`
# parameter of the constructor that determines if a bounding box is rotated or
# not. In this instance, we use the
# :attr:`~torchvision.tv_tensors.BoundingBoxFormat` kind `CXCYWHR`. The first
# two values are the `x` and `y` coordinates of the center of the bounding box.
# The next two values are the `width` and `height` of the bounding box, and the
# last value is the `rotation` of the bounding box.


orig_box = tv_tensors.BoundingBoxes(
    [
        [860.0, 1100, 570, 1840, -7],
    ],
    format="CXCYWHR",
    canvas_size=(orig_img.size[1], orig_img.size[0]),
)

plot([(orig_img, orig_box)], bbox_width=10)

# %%
# Rotation
# --------
# Rotated bounding boxes maintain their rotation with respect to the image even
# when the image itself is rotated through the
# :class:`~torchvision.transforms.RandomRotation` transform.
rotater = v2.RandomRotation(degrees=(0, 180), expand=True)
rotated_imgs = [rotater((orig_img, orig_box)) for _ in range(4)]
plot([(orig_img, orig_box)] + rotated_imgs, bbox_width=10)

# %%
# Padding
# -------
# Rotated bounding boxes also maintain their properties when the image is padded using
# :class:`~torchvision.transforms.Pad`.
padded_imgs_and_boxes = [
    v2.Pad(padding=padding)(orig_img, orig_box)
    for padding in (30, 50, 100, 200)
]
plot([(orig_img, orig_box)] + padded_imgs_and_boxes, bbox_width=10)

# %%
# Resizing
# --------
# Rotated bounding boxes are also resized along with an image in the
# :class:`~torchvision.transforms.Resize` transform.
#
# Note that the bounding box looking bigger in the images with less pixels is
# an artifact, not reality. That is merely the rasterised representation of the
# bounding box's boundaries appearing bigger because we specify a fixed width of
# that rasterized line. When the image is, say, only 30 pixels wide, a
# line that is 3 pixels wide is relatively large.
resized_imgs = [
    v2.Resize(size=size)(orig_img, orig_box)
    for size in (30, 50, 100, orig_img.size)
]
plot([(orig_img, orig_box)] + resized_imgs, bbox_width=5)

# %%
# Perspective
# -----------
# The rotated bounding box is also transformed along with the image when the
# perspective is transformed with :class:`~torchvision.transforms.RandomPerspective`.
perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_imgs = [perspective_transformer(orig_img, orig_box) for _ in range(4)]
plot([(orig_img, orig_box)] + perspective_imgs, bbox_width=10)

# %%
# Elastic Transform
# -----------------
# The rotated bounding box is appropriately unchanged when going through the
# :class:`~torchvision.transforms.ElasticTransform`.
elastic_imgs = [
    v2.ElasticTransform(alpha=alpha)(orig_img, orig_box)
    for alpha in (100.0, 500.0, 1000.0, 2000.0)
]
plot([(orig_img, orig_box)] + elastic_imgs, bbox_width=10)

# %%
# Crop & Clamping Modes
# ---------------------
# The :class:`~torchvision.transforms.CenterCrop` transform selectively crops
# the image on a center location. The behavior of the rotated bounding box
# depends on its `clamping_mode`. We can set the `clamping_mode` in the
# :class:`~torchvision.tv_tensors.BoundingBoxes` constructur, or by directly
# setting it after construction as we do in the example below.
#
# There are two values for `clamping_mode`:
#
#  - `"soft"`: The default when constucting
#    :class:`~torchvision.tv_tensors.BoundingBoxes`. <Insert semantic
#    description for soft mode.>
#  - `"hard"`: <Insert semantic description for hard mode.>
#
# For standard bounding boxes, both modes behave the same. We also need to
# document:
#
#  - `clamping_mode` for individual kernels.
#  - `clamping_mode` in :class:`~torchvision.transforms.v2.ClampBoundingBoxes`.
#  - the new :class:`~torchvision.transforms.v2.SetClampingMode` transform.
#
assert orig_box.clamping_mode == "soft"
hard_box = orig_box.clone()
hard_box.clamping_mode = "hard"

soft_center_crops_and_boxes = [
    v2.CenterCrop(size=size)(orig_img, orig_box)
    for size in (800, 1200, 2000, orig_img.size)
]

hard_center_crops_and_boxes = [
    v2.CenterCrop(size=size)(orig_img, hard_box)
    for size in (800, 1200, 2000, orig_img.size)
]

plot([[(orig_img, orig_box)] + soft_center_crops_and_boxes,
      [(orig_img, hard_box)] + hard_center_crops_and_boxes],
     bbox_width=10)
