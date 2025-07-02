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
    clamping_mode="hard",
)
# TODO: why is this necessary?
orig_box = v2.ConvertBoundingBoxFormat("xyxyxyxy")(orig_box)

plot([(orig_img, orig_box)])

# %%
# Image Rotation
# ---------------
# We can rotate the image itself, and the already rotated bounding boxes are
# rotated appropriately.

out_img, out_box = v2.RandomRotation(degrees=(0, 180), expand=True)(orig_img, orig_box)
plot([(out_img, out_box)])

# %%
# Image Padding
# -------------
# The rotated bounding boxes also respect padding transforms.
padded_imgs_and_boxes = [
    v2.Pad(padding=padding)(orig_img, orig_box) for padding in (10, 30, 50, 100)
]
plot([(orig_img, orig_box)] + padded_imgs_and_boxes)

# %%
# Image Resizing
# --------------
# The rotated bounding boxes are resized along with the image.
resized_imgs_and_boxes = [v2.Resize(size=size)(orig_img, orig_box) for size in (30, 50, 100, orig_img.size)]
plot([(orig_img, orig_box)] + resized_imgs_and_boxes)

# %%
# Image Rotation
# --------------
rotater = v2.RandomRotation(degrees=(0, 180))
rotated_imgs = [rotater((orig_img, orig_box)) for _ in range(4)]
plot([(orig_img, orig_box)] + rotated_imgs)

# %%
# Elastic Transform
# -----------------
plot([v2.ElasticTransform(alpha=250.0)(orig_img, orig_box)])

# %%
# Clamping Modes
# --------------
# Explain hard and soft, with appropriate links to documentation. Talk about
# defaults. Link to to-be-written-tutorial on mode-setting in general.
soft_box = orig_box.clone()
soft_box.clamping_mode = "soft"

hard_center_crops_and_boxes = [
    v2.CenterCrop(size=size)(orig_img, orig_box)
    for size in (800, 1200, 2000, orig_img.size)
]

soft_center_crops_and_boxes = [
    v2.CenterCrop(size=size)(orig_img, soft_box)
    for size in (800, 1200, 2000, orig_img.size)
]

plot([[(orig_img, orig_box)] + hard_center_crops_and_boxes,
      [(orig_img, soft_box)] + soft_center_crops_and_boxes])
