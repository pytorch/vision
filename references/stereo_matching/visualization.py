from typing import List
import torch
import numpy as np
from torch import Tensor
from torchvision.utils import make_grid

@torch.no_grad()
def make_disparity_image(disparity: Tensor):
    # normalize image to [0, 1]
    disparity = disparity.detach().cpu()
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    return disparity

@torch.no_grad()
def make_disparity_image_pairs(disparity: Tensor, image: Tensor):
    disparity = make_disparity_image(disparity)
    # image is in [-1, 1], bring it to [0, 1]
    image = image.detach().cpu()
    image = image * 0.5 + 0.5
    return disparity, image

@torch.no_grad()
def make_disparity_sequence(disparities: List[Tensor]):
    # convert each disparity to [0, 1]
    for idx, disparity_batch in enumerate(disparities):
        disparities[idx] = torch.stack(list(map(make_disparity_image, disparity_batch)))
    # make the list into a batch
    disparity_sequences = torch.stack(disparities)
    return disparity_sequences

@torch.no_grad()
def make_pair_grid(*inputs, orientation="horizontal"):
    # make a grid of images with the outputs and references side by side
    if orientation == "horizontal":
        # interleave the outputs and references
        canvas = torch.zeros_like(inputs[0])
        canvas = torch.cat([canvas] * len(inputs), dim=0)
        size = len(inputs)
        for idx, inp in enumerate(inputs):
            canvas[idx::size, ...] = inp
        grid = make_grid(canvas, nrow=len(inputs), padding=16, normalize=True, scale_each=True)
    elif orientation == "vertical":
        # interleave the outputs and references
        canvas = torch.cat(inputs, dim=0)
        size = len(inputs)
        for idx, inp in enumerate(inputs):
            canvas[idx::size, ...] = inp
        grid = make_grid(canvas, nrow=len(inputs[0]), padding=16, normalize=True, scale_each=True)
    else:
        raise ValueError("Unknown orientation: {}".format(orientation))
    return grid


    