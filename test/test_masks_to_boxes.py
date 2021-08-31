import os.path

import PIL.Image
import numpy
import pytest
import torch

import torchvision.ops

ASSETS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_masks_to_boxes():
    with PIL.Image.open(os.path.join(ASSETS_DIRECTORY, "masks.tiff")) as image:
        frames = numpy.zeros((image.n_frames, image.height, image.width), numpy.int)

        for index in range(image.n_frames):
            image.seek(index)

            frames[index] = numpy.array(image)

        masks = torch.tensor(frames)

    expected = torch.tensor(
        [[127, 2, 165, 40],
         [4, 100, 88, 184],
         [168, 189, 294, 300],
         [556, 272, 700, 416],
         [800, 560, 990, 725],
         [294, 828, 594, 1092],
         [756, 1036, 1064, 1491]]
    )

    torch.testing.assert_close(torchvision.ops.masks_to_boxes(masks), expected)
