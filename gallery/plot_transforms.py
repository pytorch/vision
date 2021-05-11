"""
==========================
Illustration of transforms
==========================

This example illustrates the various transforms available in :ref:`the
torchvision.transforms module <transforms>`.
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
# Pad
# ---
# The :class:`~torchvision.transforms.Pad` transform
# (see also :func:`~torchvision.transforms.functional.pad`)
# fills image borders with some pixel values.
padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
plot(padded_imgs)

####################################
# Resize
# ------
# The :class:`~torchvision.transforms.Resize` transform
# (see also :func:`~torchvision.transforms.functional.resize`)
# resizes an image.
resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
plot(resized_imgs)

####################################
# CenterCrop
# ----------
# The :class:`~torchvision.transforms.CenterCrop` transform
# (see also :func:`~torchvision.transforms.functional.center_crop`)
# crops the given image at the center.
center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
plot(center_crops)

####################################
# FiveCrop
# --------
# The :class:`~torchvision.transforms.FiveCrop` transform
# (see also :func:`~torchvision.transforms.functional.five_crop`)
# crops the given image into four corners and the central crop.
(top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(100, 100))(orig_img)
plot([top_left, top_right, bottom_left, bottom_right, center])

####################################
# Grayscale
# ---------
# The :class:`~torchvision.transforms.Grayscale` transform
# (see also :func:`~torchvision.transforms.functional.to_grayscale`)
# converts an image to grayscale
gray_img = T.Grayscale()(orig_img)
plot([gray_img], cmap='gray')

####################################
# Random transforms
# -----------------
# The following transforms are random, which means that the same transfomer
# instance will produce different result each time it transforms a given image.
#
# ColorJitter
# ~~~~~~~~~~~
# The :class:`~torchvision.transforms.ColorJitter` transform
# randomly changes the brightness, saturation, and other properties of an image.
jitter = T.ColorJitter(brightness=.5, hue=.3)
jitted_imgs = [jitter(orig_img) for _ in range(4)]
plot(jitted_imgs)

####################################
# GaussianBlur
# ~~~~~~~~~~~~
# The :class:`~torchvision.transforms.GaussianBlur` transform
# (see also :func:`~torchvision.transforms.functional.gaussian_blur`)
# performs gaussian blur transform on an image.
blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
blurred_imgs = [blurrer(orig_img) for _ in range(4)]
plot(blurred_imgs)

####################################
# RandomPerspective
# ~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomPerspective` transform
# (see also :func:`~torchvision.transforms.functional.perspective`)
# performs random perspective transform on an image.
perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
plot(perspective_imgs)

####################################
# RandomRotation
# ~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomRotation` transform
# (see also :func:`~torchvision.transforms.functional.rotate`)
# rotates an image with random angle.
rotater = T.RandomRotation(degrees=(0, 180))
rotated_imgs = [rotater(orig_img) for _ in range(4)]
plot(rotated_imgs)

####################################
# RandomAffine
# ~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomAffine` transform
# (see also :func:`~torchvision.transforms.functional.affine`)
# performs random affine transform on an image.
affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
affine_imgs = [affine_transfomer(orig_img) for _ in range(4)]
plot(affine_imgs)

####################################
# RandomCrop
# ~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomCrop` transform
# (see also :func:`~torchvision.transforms.functional.crop`)
# crops an image at a random location.
cropper = T.RandomCrop(size=(128, 128))
crops = [cropper(orig_img) for _ in range(4)]
plot(crops)

####################################
# RandomResizedCrop
# ~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomResizedCrop` transform
# (see also :func:`~torchvision.transforms.functional.resized_crop`)
# crops an image at a random location, and then resizes the crop to a given
# size.
resize_cropper = T.RandomResizedCrop(size=(32, 32))
resized_crops = [resize_cropper(orig_img) for _ in range(4)]
plot(resized_crops)

####################################
# RandomInvert
# ~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomInvert` transform
# (see also :func:`~torchvision.transforms.functional.invert`)
# randomly inverts the colors of the given image.
inverter = T.RandomInvert()
invertered_imgs = [inverter(orig_img) for _ in range(4)]
plot(invertered_imgs)

####################################
# RandomPosterize
# ~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomPosterize` transform
# (see also :func:`~torchvision.transforms.functional.posterize`)
# randomly posterizes the image by reducing the number of bits
# of each color channel.
posterizer = T.RandomPosterize(bits=2)
posterized_imgs = [posterizer(orig_img) for _ in range(4)]
plot(posterized_imgs)

####################################
# RandomSolarize
# ~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomSolarize` transform
# (see also :func:`~torchvision.transforms.functional.solarize`)
# randomly solarizes the image by inverting all pixel values above
# the threshold.
solarizer = T.RandomSolarize(threshold=192.0)
solarized_imgs = [solarizer(orig_img) for _ in range(4)]
plot(solarized_imgs)

####################################
# RandomAdjustSharpness
# ~~~~~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomAdjustSharpness` transform
# (see also :func:`~torchvision.transforms.functional.adjust_sharpness`)
# randomly adjusts the sharpness of the given image.
sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]
plot(sharpened_imgs)

####################################
# RandomAutocontrast
# ~~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomAutocontrast` transform
# (see also :func:`~torchvision.transforms.functional.autocontrast`)
# randomly applies autocontrast to the given image.
autocontraster = T.RandomAutocontrast()
autocontrasted_imgs = [autocontraster(orig_img) for _ in range(4)]
plot(autocontrasted_imgs)

####################################
# RandomEqualize
# ~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomEqualize` transform
# (see also :func:`~torchvision.transforms.functional.equalize`)
# randomly equalizes the histogram of the given image.
equalizer = T.RandomEqualize()
equalized_imgs = [equalizer(orig_img) for _ in range(4)]
plot(equalized_imgs)

####################################
# AutoAugment
# ~~~~~~~~~~~
# The :class:`~torchvision.transforms.AutoAugment` transform
# automatically augments data based on a given auto-augmentation policy.
# See :class:`~torchvision.transforms.AutoAugmentPolicy` for the available policies.
policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
augmenters = [T.AutoAugment(policy) for policy in policies]
imgs = [
    [augmenter(orig_img) for _ in range(4)]
    for augmenter in augmenters
]
row_title = [str(policy).split('.')[-1] for policy in policies]
plot(imgs, row_title=row_title)

####################################
# Randomly-applied transforms
# ---------------------------
#
# Some transforms are randomly-applied given a probability ``p``.  That is, the
# transformed image may actually be the same as the original one, even when
# called with the same transformer instance!
#
# RandomHorizontalFlip
# ~~~~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomHorizontalFlip` transform
# (see also :func:`~torchvision.transforms.functional.hflip`)
# performs horizontal flip of an image, with a given probability.
hflipper = T.RandomHorizontalFlip(p=0.5)
transformed_imgs = [hflipper(orig_img) for _ in range(4)]
plot(transformed_imgs)

####################################
# RandomVerticalFlip
# ~~~~~~~~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomVerticalFlip` transform
# (see also :func:`~torchvision.transforms.functional.vflip`)
# performs vertical flip of an image, with a given probability.
vflipper = T.RandomVerticalFlip(p=0.5)
transformed_imgs = [vflipper(orig_img) for _ in range(4)]
plot(transformed_imgs)

####################################
# RandomApply
# ~~~~~~~~~~~
# The :class:`~torchvision.transforms.RandomApply` transform
# randomly applies a list of transforms, with a given probability.
applier = T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)
transformed_imgs = [applier(orig_img) for _ in range(4)]
plot(transformed_imgs)
