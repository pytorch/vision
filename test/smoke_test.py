"""Run smoke tests"""

import os

import torchvision
from torchvision.io import read_image

image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
)
print("torchvision version is ", torchvision.__version__)
img = read_image(image_path)
