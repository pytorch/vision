"""
=======================
Repurposing annotations
=======================

The following example illustrates the operations available in the torchvision.ops module for repurposing object
localization annotations for different tasks (e.g. transforming masks used by instance and panoptic segmentation
methods into bounding boxes used by object detection methods).
"""
import os

from PIL import Image
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes

ASSETS_DIRECTORY = "../assets"

matplotlib.pyplot.rcParams["savefig.bbox"] = "tight"


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
#
# Converting Segmentation Masks to bounding boxes
# -----------------------------------------------
# For example, the masks to bounding_boxes operation can be used to transform masks into bounding boxes that can be
# used as input to detection models such as FasterRCNN and RetinaNet.

# Let's visualize an image and segmentation masks. We will use the plotting utilities provided by torchvision.utils
# to visualize. The images are taken from PenFudan Dataset.
#

with PIL.Image.open(os.path.join(ASSETS_DIRECTORY, "masks.tiff")) as image:
    masks = torch.zeros((image.n_frames, image.height, image.width), dtype=torch.int)

    for index in range(image.n_frames):
        image.seek(index)

        frame = np.array(image)

        masks[index] = torch.tensor(frame)

# We will use the masks_to_boxes from the torchvision.ops module
# It returns the boxes in (xmin, ymin, xmax, ymax) format.

bounding_boxes = masks_to_boxes(masks)

# These can be visualized very easily with draw_bounding_boxes utility
# provided in torchvision.utils.

figure = matplotlib.pyplot.figure()

a = figure.add_subplot(121)
b = figure.add_subplot(122)

labeled_image = torch.sum(masks, 0)

a.imshow(labeled_image)
b.imshow(labeled_image)

for bounding_box in bounding_boxes:
    x0, y0, x1, y1 = bounding_box

    rectangle = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none")

    b.add_patch(rectangle)

a.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
b.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# Converting Segmentation Dataset to Detection Dataset
# ----------------------------------------------------
#
# With this utility it becomes very simple to convert a segmentation dataset to a detection dataset.
# We will also use box_convert method from torchvision ops which will help in converting boxes to desired format.
# One can similarly convert panoptic dataset to detection dataset.
