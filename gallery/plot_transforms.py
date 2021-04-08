"""
==========================
Illustration of transforms
==========================

This file illustrates how torchvision transforms work on some images.

(Note: this is super WIP)
"""

####################################
# Pad
# ---
#
# The :class:`~torchvision.transforms.Pad` transform is super cool

# from PIL import Image
# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4])
from PIL import Image, ImageMath
from skimage.data import astronaut
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Pad

img = Image.fromarray(astronaut())

plt.figure().suptitle("Before padding")
plt.imshow(np.asarray(img))

padded_img = Pad(padding=30)(img)
plt.figure().suptitle("After padding")
plt.imshow(np.asarray(padded_img))

####################################
# Resize 
# ------
#
# The :class:`~torchvision.transforms.Resize` transform is even cooler

from torchvision.transforms import Resize
plt.figure().suptitle("Before resize")
plt.imshow(np.asarray(img))

padded_img = Resize(size=30)(img)
plt.figure().suptitle("After resize")
plt.imshow(np.asarray(padded_img))


print(np.exp(3))