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
        img = img.detach()
        img = F.to_pil_image(img)
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
# image. We can set the colors, labels, width as well as font and font size.
# The boxes are in ``(xmin, ymin, xmax, ymax)`` format.

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
# draw segmentation amasks on images.
#
# We will see how to use it with torchvision's FCN Resnet-50, loaded with
# :func:`~torchvision.models.segmentation.fcn_resnet50`.
# You can also try using
# DeepLabv3 (:func:`~torchvision.models.segmentation.deeplabv3_resnet50`)
# or lraspp mobilenet models
# (:func:`~torchvision.models.segmentation.lraspp_mobilenet_v3_large`).
#
# Let's start by looking at the ouput of the model. Remember that in general,
# images must be normalized before they're passed to the model.

from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import draw_segmentation_masks


model = fcn_resnet50(pretrained=True, progress=False)
model = model.eval()

normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
output = model(normalized_batch)['out']
print(output.shape, output.min().item(), output.max().item())

#####################################
# As we can see above, the output of the segmentation model is a tensor of shape
# ``(batch_size, num_classes, H, W)``. Each value is a non-normalized score
# and can normalize them into ``[0, 1]`` by using a softmax. After the softmax,
# we can interpret each value as a probability indicating how likely a given
# pixel is to belong to a given class. 
# 
# Let's plot the masks that have been detected for the dog class and for the
# boat class:

seg_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
seg_class_to_idx = {cls: idx for (idx, cls) in enumerate(seg_classes)}

# We normalize the masks of each image in the batch independently
normalized_masks = torch.stack([torch.nn.Softmax(dim=0)(masks) for masks in output])

dog_and_boat_masks = [
    normalized_masks[img_idx, seg_class_to_idx[cls]]
    for img_idx in range(batch.shape[0])
    for cls in ('dog', 'boat')
]

show(dog_and_boat_masks)

#####################################
# As expected, the model is confident about the dog class, but not so much for
# the boat class.
#
# The :func:`~torchvision.utils.draw_segmentation_masks` function can be used to
# plots those masks on top of the original image. This function expects the
# masks to be boolean masks, but our masks above contain probabilities in ``[0,
# 1]``. To get boolean masks, we can do the following:

# dogs_with_dog_masks = [
#     draw_segmentation_masks(dog_int, masks=output[img_idx, seg_class_to_idx['dog']], alpha=0.6)
#     for img_idx, dog_int in enumerate(dog1_int, dog2_int)
# ]
# show(dogs_with_masks)
