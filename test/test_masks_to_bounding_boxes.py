import os.path

import PIL.Image
import numpy
import pytest
import torch

ASSETS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@pytest.fixture
def labeled_image() -> torch.Tensor:
    with PIL.Image.open(os.path.join(ASSETS_DIRECTORY, "labeled_image.png")) as image:
        return torch.tensor(numpy.array(image, numpy.int))


@pytest.fixture
def masks() -> torch.Tensor:
    with PIL.Image.open(os.path.join(ASSETS_DIRECTORY, "masks.tiff")) as image:
        frames = numpy.zeros((image.n_frames, image.height, image.width), numpy.int)

        for index in range(image.n_frames):
            image.seek(index)

            frames[index] = numpy.array(image)

        return torch.tensor(frames)


def test_masks_to_bounding_boxes(masks):
    pass
