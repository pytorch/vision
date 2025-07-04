"""
===============================================================
Transforms on Rotated Bounding Boxes
===============================================================

Introduction.

First, some code to set everything up.
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
# Rotated Bounding Boxes
# ----------------------
# Brief intro into what rotated bounding boxes are. Brief description of the
# image.

orig_box = tv_tensors.BoundingBoxes(
    [
        [860.0, 1100, 570, 1840, -7],
    ],
    format="CXCYWHR",
    canvas_size=(orig_img.size[1], orig_img.size[0]),
)

# Note that this conversion is only necessary because our utilities for drawing
# the bounding boxes expect this format. This conversion is not necessary in
# actual training pipelines. TODO: verify with @NicolasHug, @AntoineSimoulin
orig_box = v2.ConvertBoundingBoxFormat("xyxyxyxy")(orig_box)

plot([(orig_img, orig_box)], bbox_width=10)

# %%
# Rotation
# --------------
rotater = v2.RandomRotation(degrees=(0, 180), expand=True)
rotated_imgs = [rotater((orig_img, orig_box)) for _ in range(4)]
plot([(orig_img, orig_box)] + rotated_imgs, bbox_width=10)

# %%
# Padding
# -------------
# The rotated bounding boxes also respect padding transforms.
padded_imgs_and_boxes = [
    v2.Pad(padding=padding)(orig_img, orig_box)
    for padding in (30, 50, 100, 200)
]
plot([(orig_img, orig_box)] + padded_imgs_and_boxes, bbox_width=10)

# %%
# Resizing
# --------------
# Note that the bounding box looking bigger in the small images is an artifact,
# not reality. It is due to the the fact that we specify a fixed-size for the
# width of the lines to draw. When the image is, say, only 30 pixels wide, a
# line that is 3 pixels wide is relatively large. We could potentially try to
# tweak the plotting function to avoid this appearance.
resized_imgs = [
    v2.Resize(size=size)(orig_img, orig_box)
    for size in (30, 50, 100, orig_img.size)
]
plot([(orig_img, orig_box)] + resized_imgs, bbox_width=3)

# %%
# Perspective
# -----------
perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_imgs = [perspective_transformer(orig_img, orig_box) for _ in range(4)]
plot([(orig_img, orig_box)] + perspective_imgs, bbox_width=10)

# %%
# Elastic Transform
# -----------------
elastic_imgs = [
    v2.ElasticTransform(alpha=alpha)(orig_img, orig_box)
    for alpha in (100.0, 500.0, 1000.0, 2000.0)
]
plot([(orig_img, orig_box)] + elastic_imgs, bbox_width=10)

# %%
# Crop & Clamping Modes
# ---------------------
# This section doubles as the example for Crop and for explaining clamping
# modes. My rationale for doing at both at once: any meaningful examples for
# cropping are going to impact the bounding box, and the only way to make
# sense of that is to also explain clamping modes. We should cover:
#
# * Clamping mode kinds: hard, soft, None. Behavior of each, when to use them.
# * Clamping mode defaults for: bounding boxes, functionals, transforms in
#   general, ClampingBoundingBoxes() specifically.
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
