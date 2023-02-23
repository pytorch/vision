"""
==================================
Quick-start with Transforms API v2
==================================

This example illustrates how to update existing user's data augmentations pipeline with
new Torchvision Transforms API v2 (:ref:`image transformations <transforms>`).

"""

#######################################
# Classification pipeline
# -----------------------
# For the classification task, let us take as example ImageNet
# training data augmentation pipeline (TODO: add a link to ref script).
# The only change needed is to replace the imported module `transforms` by `v2`:

import torch
from torchvision.transforms import autoaugment
from torchvision.transforms import InterpolationMode

mode = InterpolationMode.BILINEAR
aa_policy = autoaugment.AutoAugmentPolicy("imagenet")
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#######################################
#
#   .. code-block:: diff
#
#       - from torchvision.transforms import transforms
#       + from torchvision.transforms import v2 as transforms
#
#
#       t = transforms.Compose([
#           transforms.RandomResizedCrop(224, interpolation=mode),
#           transforms.RandomHorizontalFlip(),
#       -   autoaugment.AutoAugment(policy=aa_policy, interpolation=mode),
#       +   transforms.AutoAugment(policy=aa_policy, interpolation=mode),
#           transforms.PILToTensor(),  # optionally, we can use transforms.ToImageTensor() instead
#           transforms.ConvertImageDtype(torch.float),
#           transforms.Normalize(mean=mean, std=std),
#           transforms.RandomErasing()
#       ])
#

#######################################
# Object detection pipeline
# -------------------------
# TODO



