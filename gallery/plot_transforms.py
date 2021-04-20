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
# performs random perspective transform on an image.
perspectived_img = T.RandomPerspective(distortion_scale=0.6, p=1.0)(orig_img)
plot(perspectived_img, "Perspective transformed image")

####################################
# RandomRotation
# --------------
# The :class:`~torchvision.transforms.RandomRotation` transform
# rotates an image with random angle.
rotated_img = T.RandomRotation(degrees=(30, 70))(orig_img)
plot(rotated_img, "Rotated image")

####################################
# RandomAffine
# ------------
# The :class:`~torchvision.transforms.RandomAffine` transform
# performs random affine transform on an image.
affined_img = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))(orig_img)
plot(affined_img, "Affine transformed image")
