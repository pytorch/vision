"""
=======================
Visualization utilities
=======================

This example illustrates some of the utilities that torchvision offers for
visualizing images, bounding boxes, and segmentation masks.
"""


import torch
import numpy as np
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
# :func:`~torchvision.models.detection.retinanet_resnet50_fpn`, an SSDlite with
# :func:`~torchvision.models.detection.ssdlite320_mobilenet_v3_large` or an SSD with
# :func:`~torchvision.models.detection.ssd300_vgg16`. For more details
# on the output of such models, you may refer to :ref:`instance_seg_output`.

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype


batch_int = torch.stack([dog1_int, dog2_int])
batch = convert_image_dtype(batch_int, dtype=torch.float)

model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

outputs = model(batch)
print(outputs)

#####################################
# Let's plot the boxes detected by our model. We will only plot the boxes with a
# score greater than a given threshold.

score_threshold = .8
dogs_with_boxes = [
    draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
    for dog_int, output in zip(batch_int, outputs)
]
show(dogs_with_boxes)

#####################################
# Visualizing segmentation masks
# ------------------------------
# The :func:`~torchvision.utils.draw_segmentation_masks` function can be used to
# draw segmentation masks on images. Semantic segmentation and instance
# segmentation models have different outputs, so we will treat each
# independently.
#
# .. _semantic_seg_output:
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
# images must be normalized before they're passed to a semantic segmentation
# model.

from torchvision.models.segmentation import fcn_resnet50


model = fcn_resnet50(pretrained=True, progress=False)
model = model.eval()

normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
output = model(normalized_batch)['out']
print(output.shape, output.min().item(), output.max().item())

#####################################
# As we can see above, the output of the segmentation model is a tensor of shape
# ``(batch_size, num_classes, H, W)``. Each value is a non-normalized score, and
# we can normalize them into ``[0, 1]`` by using a softmax. After the softmax,
# we can interpret each value as a probability indicating how likely a given
# pixel is to belong to a given class.
#
# Let's plot the masks that have been detected for the dog class and for the
# boat class:

sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

normalized_masks = torch.nn.functional.softmax(output, dim=1)

dog_and_boat_masks = [
    normalized_masks[img_idx, sem_class_to_idx[cls]]
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
boolean_dog_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['dog'])
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
#   directly (as the softmax operation preserves the order).
#
# Now that we have boolean masks, we can use them with
# :func:`~torchvision.utils.draw_segmentation_masks` to plot them on top of the
# original images:

from torchvision.utils import draw_segmentation_masks

dogs_with_masks = [
    draw_segmentation_masks(img, masks=mask, alpha=0.7)
    for img, mask in zip(batch_int, boolean_dog_masks)
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

dog_with_all_masks = draw_segmentation_masks(dog1_int, masks=dog1_all_classes_masks, alpha=.6)
show(dog_with_all_masks)

#####################################
# We can see in the image above that only 2 masks were drawn: the mask for the
# background and the mask for the dog. This is because the model thinks that
# only these 2 classes are the most likely ones across all the pixels. If the
# model had detected another class as the most likely among other pixels, we
# would have seen its mask above.
#
# Removing the background mask is as simple as passing
# ``masks=dog1_all_classes_masks[1:]``, because the background class is the
# class with index 0.
#
# Let's now do the same but for an entire batch of images. The code is similar
# but involves a bit more juggling with the dimensions.

class_dim = 1
all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
print(f"shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}")
# The first dimension is the classes now, so we need to swap it
all_classes_masks = all_classes_masks.swapaxes(0, 1)

dogs_with_masks = [
    draw_segmentation_masks(img, masks=mask, alpha=.6)
    for img, mask in zip(batch_int, all_classes_masks)
]
show(dogs_with_masks)


#####################################
# .. _instance_seg_output:
#
# Instance segmentation models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instance segmentation models have a significantly different output from the
# semantic segmentation models. We will see here how to plot the masks for such
# models. Let's start by analyzing the output of a Mask-RCNN model. Note that
# these models don't require the images to be normalized, so we don't need to
# use the normalized batch.
#
# .. note::
#
#     We will here describe the output of a Mask-RCNN model. The models in
#     :ref:`object_det_inst_seg_pers_keypoint_det` all have a similar output
#     format, but some of them may have extra info like keypoints for
#     :func:`~torchvision.models.detection.keypointrcnn_resnet50_fpn`, and some
#     of them may not have masks, like
#     :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn`.

from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

output = model(batch)
print(output)

#####################################
# Let's break this down. For each image in the batch, the model outputs some
# detections (or instances). The number of detections varies for each input
# image. Each instance is described by its bounding box, its label, its score
# and its mask.
#
# The way the output is organized is as follows: the output is a list of length
# ``batch_size``. Each entry in the list corresponds to an input image, and it
# is a dict with keys 'boxes', 'labels', 'scores', and 'masks'. Each value
# associated to those keys has ``num_instances`` elements in it.  In our case
# above there are 3 instances detected in the first image, and 2 instances in
# the second one.
#
# The boxes can be plotted with :func:`~torchvision.utils.draw_bounding_boxes`
# as above, but here we're more interested in the masks. These masks are quite
# different from the masks that we saw above for the semantic segmentation
# models.

dog1_output = output[0]
dog1_masks = dog1_output['masks']
print(f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
      f"min = {dog1_masks.min()}, max = {dog1_masks.max()}")

#####################################
# Here the masks corresponds to probabilities indicating, for each pixel, how
# likely it is to belong to the predicted label of that instance. Those
# predicted labels correspond to the 'labels' element in the same output dict.
# Let's see which labels were predicted for the instances of the first image.

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

print("For the first dog, the following instances were detected:")
print([inst_classes[label] for label in dog1_output['labels']])

#####################################
# Interestingly, the model detects two persons in the image. Let's go ahead and
# plot those masks. Since :func:`~torchvision.utils.draw_segmentation_masks`
# expects boolean masks, we need to convert those probabilities into boolean
# values. Remember that the semantic of those masks is "How likely is this pixel
# to belong to the predicted class?". As a result, a natural way of converting
# those masks into boolean values is to threshold them with the 0.5 probability
# (one could also choose a different threshold).

proba_threshold = 0.5
dog1_bool_masks = dog1_output['masks'] > proba_threshold
print(f"shape = {dog1_bool_masks.shape}, dtype = {dog1_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
dog1_bool_masks = dog1_bool_masks.squeeze(1)

show(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))

#####################################
# The model seems to have properly detected the dog, but it also confused trees
# with people. Looking more closely at the scores will help us plotting more
# relevant masks:

print(dog1_output['scores'])

#####################################
# Clearly the model is less confident about the dog detection than it is about
# the people detections. That's good news. When plotting the masks, we can ask
# for only those that have a good score. Let's use a score threshold of .75
# here, and also plot the masks of the second dog.

score_threshold = .75

boolean_masks = [
    out['masks'][out['scores'] > score_threshold] > proba_threshold
    for out in output
]

dogs_with_masks = [
    draw_segmentation_masks(img, mask.squeeze(1))
    for img, mask in zip(batch_int, boolean_masks)
]
show(dogs_with_masks)

#####################################
# The two 'people' masks in the first image where not selected because they have
# a lower score than the score threshold. Similarly in the second image, the
# instance with class 15 (which corresponds to 'bench') was not selected.
