"""
==========================
Illustration of transforms
==========================

This example illustrates the various transforms available in :mod:`torchvision.transforms`.
"""

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as T


orig_img = Image.open(Path('assets') / 'astronaut.jpg')


def plot(img, title: str = "", with_orig: bool = True, **kwargs):
    def _plot(img, title, **kwargs):
        plt.figure().suptitle(title, fontsize=25)
        plt.imshow(np.asarray(img), **kwargs)
        plt.axis('off')

    if with_orig:
        _plot(orig_img, "Original Image")
    _plot(img, title, **kwargs)


####################################
# Pad
# ---
# The :class:`~torchvision.transforms.Pad` transform
# (see also :func:`~torchvision.transforms.functional.pad`)
# fills image borders with some pixel values.
padded_img = T.Pad(padding=30)(orig_img)
plot(padded_img, "Padded Image")

####################################
# Resize
# ------
# The :class:`~torchvision.transforms.Resize` transform
# (see also :func:`~torchvision.transforms.functional.resize`)
# resizes an image.
resized_img = T.Resize(size=30)(orig_img)
plot(resized_img, "Resized Image")

####################################
# CenterCrop
# ----------
# The :class:`~torchvision.transforms.CenterCrop` transform
# (see also :func:`~torchvision.transforms.functional.center_crop`)
# crops the given image at the center.
center_cropped_img = T.CenterCrop(size=(100, 100))(orig_img)
plot(center_cropped_img, "Center Cropped Image")


####################################
# FiveCrop
# --------
# The :class:`~torchvision.transforms.FiveCrop` transform
# (see also :func:`~torchvision.transforms.functional.five_crop`)
# crops the given image into four corners and the central crop.
(img1, img2, img3, img4, img5) = T.FiveCrop(size=(100, 100))(orig_img)
plot(img1, "Top Left Corner Image")
plot(img2, "Top Right Corner Image", with_orig=False)
plot(img3, "Bottom Left Corner Image", with_orig=False)
plot(img4, "Bottom Right Corner Image", with_orig=False)
plot(img5, "Center Image", with_orig=False)

####################################
# ColorJitter
# -----------
# The :class:`~torchvision.transforms.ColorJitter` transform
# randomly changes the brightness, saturation, and other properties of an image.
jitted_img = T.ColorJitter(brightness=.5, hue=.3)(orig_img)
plot(jitted_img, "Jitted Image")

####################################
# Grayscale
# ---------
# The :class:`~torchvision.transforms.Grayscale` transform
# (see also :func:`~torchvision.transforms.functional.to_grayscale`)
# converts an image to grayscale
gray_img = T.Grayscale()(orig_img)
plot(gray_img, "Grayscale Image", cmap='gray')

####################################
# RandomPerspective
# -----------------
# The :class:`~torchvision.transforms.RandomPerspective` transform
# (see also :func:`~torchvision.transforms.functional.perspective`)
# performs random perspective transform on an image.
perspectived_img = T.RandomPerspective(distortion_scale=0.6, p=1.0)(orig_img)
plot(perspectived_img, "Perspective transformed Image")

####################################
# RandomRotation
# --------------
# The :class:`~torchvision.transforms.RandomRotation` transform
# (see also :func:`~torchvision.transforms.functional.rotate`)
# rotates an image with random angle.
rotated_img = T.RandomRotation(degrees=(30, 70))(orig_img)
plot(rotated_img, "Rotated Image")

####################################
# RandomAffine
# ------------
# The :class:`~torchvision.transforms.RandomAffine` transform
# (see also :func:`~torchvision.transforms.functional.affine`)
# performs random affine transform on an image.
affined_img = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))(orig_img)
plot(affined_img, "Affine transformed Image")

####################################
# RandomCrop
# ----------
# The :class:`~torchvision.transforms.RandomCrop` transform
# (see also :func:`~torchvision.transforms.functional.crop`)
# crops an image at a random location.
crop_img = T.RandomCrop(size=(128, 128))(orig_img)
plot(crop_img, "Random cropped Image")

####################################
# RandomResizedCrop
# -----------------
# The :class:`~torchvision.transforms.RandomResizedCrop` transform
# (see also :func:`~torchvision.transforms.functional.resized_crop`)
# crops an image at a random location, and then resizes the crop to a given
# size.
resized_crop_img = T.RandomResizedCrop(size=(32, 32))(orig_img)
plot(resized_crop_img, "Random resized cropped Image")

####################################
# RandomHorizontalFlip
# --------------------
# The :class:`~torchvision.transforms.RandomHorizontalFlip` transform
# (see also :func:`~torchvision.transforms.functional.hflip`)
# performs horizontal flip of an image, with a given probability.
#
# .. note::
#   Since the transform is applied randomly, the two images below may actually be
#   the same.
random_hflip_img = T.RandomHorizontalFlip(p=0.5)(orig_img)
plot(random_hflip_img, "Random horizontal flipped Image")

####################################
# RandomVerticalFlip
# ------------------
# The :class:`~torchvision.transforms.RandomVerticalFlip` transform
# (see also :func:`~torchvision.transforms.functional.vflip`)
# performs vertical flip of an image, with a given probability.
#
# .. note::
#   Since the transform is applied randomly, the two images below may actually be
#   the same.
random_vflip_img = T.RandomVerticalFlip(p=0.5)(orig_img)
plot(random_vflip_img, "Random vertical flipped Image")

####################################
# RandomApply
# -----------
# The :class:`~torchvision.transforms.RandomApply` transform
# randomly applies a list of transforms, with a given probability
#
# .. note::
#   Since the transform is applied randomly, the two images below may actually be
#   the same.
random_apply_img = T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)(orig_img)
plot(random_apply_img, "Random Apply transformed Image")

####################################
# GaussianBlur
# ------------
# The :class:`~torchvision.transforms.GaussianBlur` transform
# (see also :func:`~torchvision.transforms.functional.gaussian_blur`)
# performs gaussianblur transform on an image.
gaus_blur_img = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.4, 3.0))(orig_img)
plot(gaus_blur_img, "Gaussian Blurred Image")


####################################
# AutoAugment
# -----------
# The :class:`~torchvision.transforms.AutoAugment` transform
# Automatically augments data based on available AutoAugmentation Policies.
# Use :class:`torchvision.transforms.AutoAugmentPolicy` to create policies.
cifar10_policy = T.AutoAugmentPolicy.CIFAR10
imagenet_policy = T.AutoAugmentPolicy.IMAGENET
svhn_policy = T.AutoAugmentPolicy.SVHN

cifar_img1 = T.AutoAugment(cifar10_policy)(orig_img)
cifar_img2 = T.AutoAugment(cifar10_policy)(orig_img)
cifar_img3 = T.AutoAugment(cifar10_policy)(orig_img)
cifar_img4 = T.AutoAugment(cifar10_policy)(orig_img)
cifar_img5 = T.AutoAugment(cifar10_policy)(orig_img)

imagenet_img1 = T.AutoAugment(imagenet_policy)(orig_img)
imagenet_img2 = T.AutoAugment(imagenet_policy)(orig_img)
imagenet_img3 = T.AutoAugment(imagenet_policy)(orig_img)
imagenet_img4 = T.AutoAugment(imagenet_policy)(orig_img)
imagenet_img5 = T.AutoAugment(imagenet_policy)(orig_img)

svhn_img1 = T.AutoAugment(svhn_policy)(orig_img)
svhn_img2 = T.AutoAugment(svhn_policy)(orig_img)
svhn_img3 = T.AutoAugment(svhn_policy)(orig_img)
svhn_img4 = T.AutoAugment(svhn_policy)(orig_img)
svhn_img5 = T.AutoAugment(svhn_policy)(orig_img)

# Cifar10 Policy

plot(cifar_img1, "Cifar10 Transformed Image1")
plot(cifar_img2, "Cifar10 Transformed Image2", with_orig=False)
plot(cifar_img3, "Cifar10 Transformed Image3", with_orig=False)
plot(cifar_img4, "Cifar10 Transformed Image4", with_orig=False)
plot(cifar_img5, "Cifar10 Transformed Image5", with_orig=False)

# Imagenet Policy

plot(imagenet_img1, "Imagenet Transformed Image1")
plot(imagenet_img2, "Imagenet Transformed Image2", with_orig=False)
plot(imagenet_img3, "Imagenet Transformed Image3", with_orig=False)
plot(imagenet_img4, "Imagenet Transformed Image4", with_orig=False)
plot(imagenet_img5, "Imagenet Transformed Image5", with_orig=False)

# SVHN Policy

plot(svhn_img1, "SVHN Transformed Image1")
plot(svhn_img2, "SVHN Transformed Image2", with_orig=False)
plot(svhn_img3, "SVHN Transformed Image3", with_orig=False)
plot(svhn_img4, "SVHN Transformed Image4", with_orig=False)
plot(svhn_img5, "SVHN Transformed Image5", with_orig=False)
