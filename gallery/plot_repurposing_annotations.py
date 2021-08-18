"""
=======================
Repurposing annotations
=======================

The following example illustrates the operations available in :ref:`the torchvision.ops module <ops>` for repurposing
object localization annotations for different tasks (e.g. transforming masks used by instance and panoptic
segmentation methods into bounding boxes used by object detection methods).
"""

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('assets') / 'astronaut.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

####################################
# Masks
# --------------------------------------
# In tasks like instance and panoptic segmentation, masks are commonly defined, and are defined by this package,
# as a multi-dimensional array (e.g. a NumPy array or a PyTorch tensor) with the following shape:
#
#       (objects, height, width)
#
# Where objects is the number of annotated objects in the image. Each (height, width) object corresponds to exactly
# one object. For example, if your input image has the dimensions 224 x 224 and has four annotated objects the shape
# of your masks annotation has the following shape:
#
#       (4, 224, 224).
#
# A nice property of masks is that they can be easily repurposed to be used in methods to solve a variety of object
# localization tasks.
#
# Masks to bounding boxes
# ~~~~~~~~~~~~~~~~~~~~~~~
# For example, the masks to bounding_boxes operation can be used to transform masks into bounding boxes that can be
# used in methods like Faster RCNN and YOLO.
padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
plot(padded_imgs)
