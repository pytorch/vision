"""Run smoke tests"""

import torchvision
from torchvision.io import read_image

print("torchvision version is ", torchvision.__version__)
img = read_image("pytorch/vision/test/assets/encode_jpeg/grace_hopper_517x606.jpg")
