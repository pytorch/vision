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
# Semantic segmentation models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We will see how to use it with torchvision's FCN Resnet-50, loaded with
# :func:`~torchvision.models.segmentation.fcn_resnet50`.  You can also try using
# DeepLabv3 (:func:`~torchvision.models.segmentation.deeplabv3_resnet50`) or
# lraspp mobilenet models
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
# ``(batch_size, num_classes, H, W)``. Each value is a non-normalized score and
# can normalize them into ``[0, 1]`` by using a softmax. After the softmax, we
# can interpret each value as a probability indicating how likely a given pixel
# is to belong to a given class. 
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

class_dim = 1
boolean_dog_masks = (normalized_masks.argmax(class_dim) == seg_class_to_idx['dog'])
print(f"shape = {boolean_dog_masks.shape}, dtype = {boolean_dog_masks.dtype}")
show([m.float() for m in boolean_dog_masks])

#####################################
# The line above where we define ``boolean_dog_masks`` is a bit cryptic, but you
# can read it as the following query: "For which pixels is 'dog' the most likely
# class?"
# 
# .. note::
#   While we're using the ``normalized_masks`` here, we would have
#   gotten the same result by using the non-normalized scores of the model
#   directly (as the softmax operation perserves the order).
# 
# Now that we have boolean masks, we can use them with
# :func:~torchvision.utils.draw_segmentation_masks to plot them on top of the
# original images:

dogs_with_masks = [
    draw_segmentation_masks(img, masks=mask, alpha=0.3)
    for img, mask in zip(batch, boolean_dog_masks)
]
show(dogs_with_masks)

#####################################
# We can plot more than one mask per image! Remember that the model returned as
# many masks as there are classes. Let's ask the same query as above, but this
# time for *all* classes, not just the dog class: "For each pixel and each class
# C, is class C the most most likely class?"
# 
# This one is a bit more involved, so we'll first show how to do it with a
# single image, and then we'll generalize to the batch

num_classes = normalized_masks.shape[1]
dog1_masks = normalized_masks[0]
class_dim = 0
dog1_all_classes_masks = dog1_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]

print(f"dog1_masks shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}")
print(f"dog1_all_classes_masks = {dog1_all_classes_masks.shape}, dtype = {dog1_all_classes_masks.dtype}")

dog_with_all_masks = draw_segmentation_masks(dog1_float, masks=dog1_all_classes_masks, alpha=.4)
show(dog_with_all_masks)

#####################################
# We can see in the image above that only 2 masks were drawn: the mask for the
# background and the mask for the dog. This is because the model thinkgs that
# only these 2 classes are the most likely ones across all the pixels. It the
# model had detected another class as the most likely among other pixels, we
# would have seen its mask above.
# 
# Removing the background mask is as simple as passing
# ``masks=dog1_all_classes_masks[1:]``.
# 
# Let's now do the same but for an entire batch of images. The code is similar
# but involves a bit more juggling with the dimensions.

class_dim = 1
all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
print(f"shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}")
# The first dimension is the classes now, so we need to swap it
all_classes_masks = all_classes_masks.swapaxes(0, 1)

dogs_with_masks = [
    draw_segmentation_masks(img, masks=mask, alpha=.4)
    for img, mask in zip(batch, all_classes_masks)
]
show(dogs_with_masks)


#####################################
# Instance segmentation models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instance segmentation models have a significantly different output from the
# semantic segmentation models. We will see here blahblah TODO
# 