"""
=======================
Repurposing annotations
=======================

The following example illustrates the operations available in :ref:`the torchvision.ops module <ops>` for repurposing
object localization annotations for different tasks (e.g. transforming masks used by instance and panoptic
segmentation methods into bounding boxes used by object detection methods).
"""

import matplotlib.patches
import matplotlib.pyplot
import numpy
import skimage.draw
import skimage.measure
import torch
import torchvision.ops.boxes


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

image, labels = skimage.draw.random_shapes((512, 256), 4, min_size=32, multichannel=False)

labeled_image = skimage.measure.label(image, background=255)

labeled_image = skimage.img_as_ubyte(labeled_image)

labeled_image = torch.tensor(labeled_image)

masks = torch.zeros((len(labels), *labeled_image.shape), dtype=torch.int)

for index in torch.unique(labeled_image):
    masks[index - 1] = torch.where(labeled_image == index, index, 0)

bounding_boxes = torchvision.ops.boxes.masks_to_boxes(torch.tensor(masks))

figure = matplotlib.pyplot.figure()

a = figure.add_subplot(121)
b = figure.add_subplot(122)

a.imshow(labeled_image)
b.imshow(labeled_image)

for bounding_box in bounding_boxes.tolist():
    x0, y0, x1, y1 = bounding_box

    rectangle = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='r', facecolor='none')

    b.add_patch(rectangle)
