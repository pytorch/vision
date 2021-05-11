"""
=======================
Visualization utilities
=======================

This example illustrates some of the utilities that torchvision offers for
visualizing images, bounding boxes, and segmentation masks.
"""


import torch
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


####################################
# Visualizing a grid of images
# ----------------------------
# The :func:`~torchvision.utils.make_grid` function can be used to create a
# tensor that represents multiple images in a grid.  This util requires a single
# image of dtype ``uint8`` as input.

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

dog1_int = read_image(str(Path('assets') / 'dog1.jpg'))
dog2_int = read_image(str(Path('assets') / 'dog2.jpg'))

grid = make_grid([dog1_int, dog2_int, dog1_int, dog2_int])
show(grid)

####################################
# Visualizing bounding boxes
# --------------------------
# We can use :func:`~torchvision.utils.draw_bounding_boxes` to draw boxes on an
# image. We can set the colors, labels, width as well as font and font size !
# The boxes are in ``(xmin, ymin, xmax, ymax)`` format
# from torchvision.utils import draw_bounding_boxes

from torchvision.utils import draw_bounding_boxes


boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
show(result)


#####################################
# Naturally, we can also plot bounding boxes produced by torchvision detection
# models.  Here is demo with a Faster R-CNN model loaded from
# :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn`
# model. You can also try using a RetinaNet with
# :func:`~torchvision.models.detection.retinanet_resnet50_fpn`.

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype


dog1_float = convert_image_dtype(dog1_int, dtype=torch.float)
dog2_float = convert_image_dtype(dog2_int, dtype=torch.float)
batch = torch.stack([dog1_float, dog2_float])

model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

outputs = model(batch)
print(outputs)

#####################################
# Let's plot the boxes detected by our model. We will only plot the boxes with a
# score greater than a given threshold.

threshold = .8
dogs_with_boxes = [
    draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > threshold], width=4)
    for dog_int, output in zip((dog1_int, dog2_int), outputs)
]
show(dogs_with_boxes)

#####################################
# Visualizing segmentation masks
# ------------------------------
# The :func:`~torchvision.utils.draw_segmentation_masks` function can be used to
# draw segmentation amasks on images. We can set the colors as well as
# transparency of masks.
#
# Here is demo with torchvision's FCN Resnet-50, loaded with
# :func:`~torchvision.models.segmentation.fcn_resnet50`.
# You can also try using
# DeepLabv3 (:func:`~torchvision.models.segmentation.deeplabv3_resnet50`)
# or lraspp mobilenet models
# (:func:`~torchvision.models.segmentation.lraspp_mobilenet_v3_large`).
#
# Like :func:`~torchvision.utils.draw_bounding_boxes`,
# :func:`~torchvision.utils.draw_segmentation_masks` requires a single RGB image
# of dtype `uint8`.

from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import draw_segmentation_masks


model = fcn_resnet50(pretrained=True, progress=False)
model = model.eval()

# The model expects the batch to be normalized
batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
outputs = model(batch)

dogs_with_masks = [
    draw_segmentation_masks(dog_int, masks=masks, alpha=0.6)
    for dog_int, masks in zip((dog1_int, dog2_int), outputs['out'])
]
show(dogs_with_masks)
