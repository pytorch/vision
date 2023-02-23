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
# For the classification task, let us take as an example
# training data augmentation pipeline for the ImageNet (`classification reference script <https://github.com/pytorch/vision/tree/96dbada4d588cabbd24ab1eee57cd261c9b93d20/references/classification>`_).
# The only change we need is to replace the imported module `transforms` by `v2`:

#######################################
#
#   .. code-block:: diff
#
#       - from torchvision.transforms import transforms as T
#       + from torchvision.transforms import v2 as T
#
#
#       t = T.Compose([
#           T.RandomResizedCrop(224, interpolation=mode),
#           T.RandomHorizontalFlip(),
#       -   autoaugment.AutoAugment(policy=aa_policy, interpolation=mode),
#       +   T.AutoAugment(policy=aa_policy, interpolation=mode),
#           T.PILToTensor(),  # optionally, we can use T.ToImageTensor() instead
#           T.ConvertImageDtype(torch.float),
#           T.Normalize(mean=mean, std=std),
#           T.RandomErasing()
#       ])
#

#######################################
# Object detection pipeline
# -------------------------
# For the object detection task, we can take the
# training data augmentation pipeline for MSCoco from our references scripts (`detection reference script <https://github.com/pytorch/vision/tree/96dbada4d588cabbd24ab1eee57cd261c9b93d20/references/detection>`_).
# For example, let us consider data augmentation policy for RetinaNet training.
# Below ``transforms`` module refers to `transforms file <https://github.com/pytorch/vision/blob/96dbada4d588cabbd24ab1eee57cd261c9b93d20/references/detection/transforms.py>`_.

#######################################
#
#   .. code-block:: diff
#
#       - import transforms as T
#       + from torchvision.transforms import v2 as T
#
#
#       t = T.Compose([
#           T.RandomShortestSize(
#               min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
#           ),
#           T.RandomHorizontalFlip(),
#           T.PILToTensor(),  # optionally, we can use T.ToImageTensor() instead
#           T.ConvertImageDtype(torch.float),
#       +   T.SanitizeBoundingBoxes(),  # explicitly remove invalid bounding boxes (and labels)
#       ])
#
# For an end-to-end object detection example using Transforms API v2, please check out :ref:`this tutorial <e2e_object_detection_v2>`.


#######################################
# Semantic segmentation pipeline
# ------------------------------
# Finally, here is how we can update the code of the training data augmentation pipeline
# for semantic image segmentation task.
# Below ``transforms`` module refers to `segmentation transforms file <https://github.com/pytorch/vision/blob/96dbada4d588cabbd24ab1eee57cd261c9b93d20/references/segmentation/transforms.py>`_.
#

#######################################
#
#   .. code-block:: diff
#
#       - import transforms as T
#       + from torchvision.transforms import v2 as T
#
#
#       t = T.Compose([
#           T.RandomResize(min_size, max_size),
#           T.RandomHorizontalFlip(),
#           T.RandomCrop(480),
#           T.PILToTensor(),  # optionally, we can use T.ToImageTensor() instead
#           T.ConvertImageDtype(torch.float),
#           T.Normalize(mean=mean, std=std),
#       ])
#





