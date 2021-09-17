import os.path

import PIL.Image
import numpy
import torch

from torchvision.ops import masks_to_boxes

ASSETS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_masks_to_boxes():
    with PIL.Image.open(os.path.join(ASSETS_DIRECTORY, "masks.tiff")) as image:
        frames = numpy.zeros((image.n_frames, image.height, image.width), int)

        for index in range(image.n_frames):
            image.seek(index)

            frames[index] = numpy.array(image)

        masks = torch.tensor(frames)

    expected = torch.tensor(
        [[127, 2, 165, 40],
         [2, 50, 44, 92],
         [56, 63, 98, 100],
         [139, 68, 175, 104],
         [160, 112, 198, 145],
         [49, 138, 99, 182],
         [108, 148, 152, 213]],
        dtype=torch.int32
    )

    torch.testing.assert_close(masks_to_boxes(masks), expected)
