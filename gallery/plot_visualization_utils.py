"""
=======================
Visualization utilities
=======================

This example illustrates some of the utilities that torchvision offers for
visualizing bounding boxes and segmentation masks.
"""


from pathlib import Path

import torch
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torchvision.io import read_image


plt.rcParams["savefig.bbox"] = 'tight'


def show(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis('off')

####################################
# Visualizing Bounding Boxes
# --------------------------
# We will draw some bounding boxes on that cute racoon:

from torchvision.utils import draw_bounding_boxes

racoon = T.ToTensor()(scipy.misc.face().copy())
racoon_int = T.ConvertImageDtype(dtype=torch.uint8)(racoon)
show(racoon_int)

#####################################
# We can use :func:`~torchvision.utils.draw_bounding_boxes` to draw boxes on an
# image. We can set the colors, labels, width as well as font and font size !
# This util requires a single image of dtype ``uint8`` as input.
# The boxes are in ``(xmin, ymin, xmax, ymax)`` format

boxes = torch.tensor([[100, 400, 500, 740], [500, 200, 800, 580]], dtype=torch.float)
labels = ["grass", "racoon"]
colors = ["blue", "yellow"]
result = draw_bounding_boxes(racoon_int, boxes, labels=labels, colors=colors, width=10)
show(result)

#####################################
# Naturally, we can also plot bounding boxes produced by torchvision detection
# models.  Here is demo with a Faster R-CNN model loaded from
# :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn`
# model. You can also try using a RetinaNet with
# :func:`~torchvision.models.detection.retinanet_resnet50_fpn`.

from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()

outputs = model(racoon.unsqueeze(0))
print(outputs)

#####################################
# Let's plot the top 5 boxes detected by our model

boxes = outputs[0]['boxes']
colors = ["blue", "red", "green", "yellow", "orange"]

result = draw_bounding_boxes(racoon_int, boxes=boxes[:5], colors=colors, width=10)
show(result)

#####################################
# Visualizing Segmentation Masks
# ------------------------------
# The :func:`~torchvision.utils.draw_segmentation_masks` function can be used to
# draw segmentation amasks on images. We can set the colors as well as
# transparency of masks.

from torchvision.utils import draw_segmentation_masks
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000281759.jpg"
umbrellas = Image.open(requests.get(url, stream=True).raw)
umbrellas = T.ToTensor()(umbrellas)
show(umbrellas)

#####################################
# Let's draw a few maks! The masks contain tensors denoting probabilites of each
# class.  Here is demo with torchvision's FCN Resnet-50, loaded with
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

model = fcn_resnet50(pretrained=True)
model = model.eval()
output = model(umbrellas.unsqueeze(0))
masks = output['out'].squeeze(0)

umbrellas_int = T.ConvertImageDtype(dtype=torch.uint8)(umbrellas)
result = draw_segmentation_masks(umbrellas_int, masks, alpha=0.2)
show(result)
