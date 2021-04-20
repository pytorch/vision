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


def plot(img, title="", with_orig=True, **kwargs):
    def _plot(img, title, **kwargs):
        plt.figure().suptitle(title, fontsize=25)
        plt.imshow(np.asarray(img), **kwargs)
        plt.axis('off')

    if with_orig:
        _plot(orig_img, "Original image")
    _plot(img, title, **kwargs)


####################################
# Pad
# ---
# The :class:`~torchvision.transforms.Pad` transform
# (see also :func:`~torchvision.transforms.functional.pad`)
# fills image borders with some pixel values.
padded_img = T.Pad(padding=30)(orig_img)
plot(padded_img, "Padded image")

####################################
# Resize
# ------
# The :class:`~torchvision.transforms.Resize` transform
# (see also :func:`~torchvision.transforms.functional.resize`)
# resizes an image.
resized_img = T.Resize(size=30)(orig_img)
plot(resized_img, "Resized image")

####################################
# ColorJitter
# -----------
# The :class:`~torchvision.transforms.ColorJitter` transform
# randomly changes the brightness, saturation, and other properties of an image.
jitted_img = T.ColorJitter(brightness=.5, hue=.3)(orig_img)
plot(jitted_img, "Jitted image")

####################################
# Grayscale
# ---------
# The :class:`~torchvision.transforms.Grayscale` transform
# (see also :func:`~torchvision.transforms.functional.to_grayscale`)
# converts an image to grayscale
gray_img = T.Grayscale()(orig_img)
plot(gray_img, "Grayscale image", cmap='gray')

####################################
# RandomPerspective
# -----------------
# The :class:`~torchvision.transforms.RandomPerspective` transform
# (see also :func:`~torchvision.transforms.functional.perspective`)
# performs random perspective transform on an image.
perspectived_img = T.RandomPerspective(distortion_scale=0.6, p=1.0)(orig_img)
plot(perspectived_img, "Perspective transformed image")

####################################
# RandomRotation
# --------------
# The :class:`~torchvision.transforms.RandomRotation` transform
# (see also :func:`~torchvision.transforms.functional.rotate`)
# rotates an image with random angle.
rotated_img = T.RandomRotation(degrees=(30, 70))(orig_img)
plot(rotated_img, "Rotated image")

####################################
# RandomAffine
# ------------
# The :class:`~torchvision.transforms.RandomAffine` transform
# (see also :func:`~torchvision.transforms.functional.affine`)
# performs random affine transform on an image.
affined_img = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))(orig_img)
plot(affined_img, "Affine transformed image")

####################################
# RandomCrop
# ----------
# The :class:`~torchvision.transforms.RandomCrop` transform
# (see also :func:`~torchvision.transforms.functional.crop`)
# performs random crop transform on an image.
random_crop_img = T.RandomCrop(size=(128, 128))(orig_img)
plot(random_crop_img, "Random Cropped transformed  image")

####################################
# RandomResizedCrop
# -----------------
# The :class:`~torchvision.transforms.RandomResizedCrop`)
# (see also :func:`~torchvision.transforms.functional.resized_crop`)
# performs random resize crop transform on an image.
rand_resize_crop_img = T.RandomResizedCrop(size=(32, 32))(orig_img)
plot(rand_resize_crop_img, "Random Resize Crop  image")

####################################
# RandomHorizontalFlip
# --------------------
# The :class:`~torchvision.transforms.RandomHorizontalFlip`)
# (see also :func:`~torchvision.transforms.functional.hflip`)
# performs random horizontal flip transform on an image.
rand_horizon_img = T.RandomHorizontalFlip(p=0.6)(orig_img)
plot(rand_horizon_img, "Random horizontal flip of  image")

####################################
# RandomVerticalFlip
# ------------------
# The :class:`~torchvision.transforms.RandomVerticalFlip`)
# (see also :func:`~torchvision.transforms.functional.vflip`)
# performs random vertical flip transform on an image.
rand_verti_img = T.RandomVerticalFlip(p=0.6)(orig_img)
plot(rand_verti_img, "Random vertical flip of  image")

####################################
# GaussianBlur
# ------------
# The :class:`~torchvision.transforms.GaussianBlur`)
# (see also :func:`~torchvision.transforms.functional.gaussian_blur`)
# performs gaussianblur transform on an image.
gaus_blur_img = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.4, 3.0))(orig_img)
plot(gaus_blur_img, "Gaussian Blur of  image")
####################################
# RandomApply
# -----------
# The :class:`~torchvision.transforms.RandomApply`)
# (see also :func:`~torchvision.transforms.functional.transforms`)
# performs random operation of transform on an image with probability.
ran_img = T.RandomApply(transforms=[T.RandomCrop(size=(8, 8))], p=0.5)(orig_img)
plot(ran_img, "Random Apply  of  image")
