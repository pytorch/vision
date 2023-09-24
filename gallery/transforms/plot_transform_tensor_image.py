"""
=======================================
Converting between tensor and PIL image
=======================================

.. note::
    Try on `collab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_custom_tv_tensors.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_custom_tv_tensors.py>` to download the full example code.

In thie example, we explain the basic usgae of :func:`~torchvision.transforms.functional.to_tensor`, :func:`~torchvision.transforms.functional.pil_to_tensor` and :func:`~torchvision.transforms.functional.to_pil_image`. 
And the difference between :func:`~torchvision.transforms.functional.to_tensor` and :func:`~torchvision.transforms.functional.pil_to_tensor`.
"""

# %%
# torchvision.transforms.functional.to_tensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this cell, we illustrate the different representation of a PIL image and tensor
import PIL.Image as Image
from torchvision.transforms.functional import to_tensor
from helpers import plot # use your favorite visualization library

img_pil = Image.open('../assets/person1.jpg')
width, height = img_pil.size
# There is no straight forward way to get channel information
# Please read https://pillow.readthedocs.io/en/stable/handbook/concepts.html for more detail
num_channels = 3 # hardcoded since it's a color image.
print("PIL image: width x height x num_channels:", width, height, num_channels)

img_tensor = to_tensor(img_pil)
num_channels, height, width = img_tensor.shape
print("Tensor image: num_channels x height x width:", num_channels, height, width)
plot([img_tensor])

# %%
# torchvision.transforms.functional.pil_to_tensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this cell, we explain the difference between :func:`~torchvision.transforms.functional.pil_to_tensor` vs. :func:`~torchvision.transforms.functional.to_tensor`
from torchvision.transforms.functional import pil_to_tensor

img_pil = Image.open('../assets/person1.jpg')
img_to_tensor = to_tensor(img_pil)
num_channels, height, width = img_to_tensor.shape
print("Tensor image(to_tensor): num_channels x height x width:", num_channels, height, width)
print("Tensor image(to_tensor) datatype:", img_to_tensor.dtype)
img_pil_to_tensor = pil_to_tensor(img_pil)
num_channels, height, width = img_pil_to_tensor.shape
print("Tensor image(pil_to_tensor): num_channels x height x width:", num_channels, height, width)
print("Tensor image(pil_to_tensor) datatype:", img_pil_to_tensor.dtype)
plot([img_to_tensor, img_pil_to_tensor])

# %%
# The shape is the same but **data type** is different! The **tensor value** is also different!

# %%
print(img_to_tensor) # tensor that is returned by calling to_tensor()
print(img_pil_to_tensor) # tensor that is returned by calling pil_to_tensor()

# %%
# Notice :func:`~torchvision.transforms.functional.to_tensor` automatically scale the image, but :func:`~torchvision.transforms.functional.pil_to_tensor` does not. To rescale the image back,

import torch
img_pil_to_tensor_2 = (img_to_tensor * 255).to(torch.uint8)
print((img_pil_to_tensor_2 == img_pil_to_tensor).all().item()) # check if two tensors are same

# %%
# **TLDR** it's recommended to use :func:`~torchvision.transforms.functional.pil_to_tensor` for visualization tasks since most visualization library
# expects input image to be ``torch.uint8``. On the other hand, :func:`~torchvision.transforms.functional.to_tensor` is better for computation tasks since models, optimizers and loss functions expect
# input image to be ``torch.float32``.


# %%
# torchvision.transforms.functional.to_pil_image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this cell, we explain an example usage of :func:`~torchvision.transforms.functional.to_pil_image`
from torchvision.transforms.functional import to_pil_image

img_pil = Image.open('../assets/person1.jpg')

# convert to tensor using to_tensor() and pil_to_tensor()
img_to_tensor = to_tensor(img_pil)
img_pil_to_tensor = pil_to_tensor(img_pil)

# convert back to PIL image from tensor
pil_img_to_tensor = to_pil_image(img_to_tensor)
pil_img_pil_to_tensor = to_pil_image(img_pil_to_tensor)
print(pil_img_to_tensor)
print(pil_img_pil_to_tensor)

# %%
# Both tensor can be converted back to PIL image.

