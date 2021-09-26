"""
=======================
Repurposing annotations
=======================

The following example illustrates the operations available in the torchvision.ops module for repurposing object
localization annotations for different tasks (e.g. transforming masks used by instance and panoptic segmentation
methods into bounding boxes used by object detection methods).
"""


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


####################################
# Masks
# -----
# In tasks like instance and panoptic segmentation, masks are commonly defined, and are defined by this package,
# as a multi-dimensional array (e.g. a NumPy array or a PyTorch tensor) with the following shape:
#
#       (num_objects, height, width)
#
# Where num_objects is the number of annotated objects in the image. Each (height, width) object corresponds to exactly
# one object. For example, if your input image has the dimensions 224 x 224 and has four annotated objects the shape
# of your masks annotation has the following shape:
#
#       (4, 224, 224).
#
# A nice property of masks is that they can be easily repurposed to be used in methods to solve a variety of object
# localization tasks.

####################################
# Converting Segmentation Masks to bounding boxes
# -----------------------------------------------
# For example, the masks to bounding_boxes operation can be used to transform masks into bounding boxes that can be
# used as input to detection models such as FasterRCNN and RetinaNet.


from torchvision.io import read_image
from torchvision.ops import masks_to_boxes

img_path = os.path.join(ASSETS_DIRECTORY, "FudanPed00054.png")
mask_path = os.path.join(ASSETS_DIRECTORY, "FudanPed00054_mask.png")
img = read_image(img_path)
mask = read_image(mask_path)
mask = F.convert_image_dtype(mask, dtype=torch.float)
obj_ids = torch.unique(mask)
obj_ids = obj_ids[1:]
masks = mask == obj_ids[:, None, None]

####################################
# Let's visualize an image and segmentation masks. We will use the plotting utilities provided by torchvision.utils
# to visualize. The images are taken from PenFudan Dataset.


from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

drawn_masks = []
for mask in masks:
    drawn_masks.append(draw_segmentation_masks(img, mask, alpha=0.8, colors="blue"))

show(drawn_masks)

####################################
# We will use the masks_to_boxes from the torchvision.ops module
# It returns the boxes in (xmin, ymin, xmax, ymax) format.

boxes = masks_to_boxes(masks)

####################################
# These can be visualized very easily with draw_bounding_boxes utility
# provided in torchvision.utils.

drawn_boxes = draw_bounding_boxes(img, boxes, colors="red")
show(drawn_boxes)

####################################
# Converting Segmentation Dataset to Detection Dataset
# ----------------------------------------------------
#
# With this utility it becomes very simple to convert a segmentation dataset to a detection dataset.
# We will also use box_convert method from torchvision ops which will help in converting boxes to desired format.
# One can similarly convert panoptic dataset to detection dataset.
