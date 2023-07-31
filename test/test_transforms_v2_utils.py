import PIL.Image
import pytest

import torch

import torchvision.transforms.v2.utils
from common_utils import make_bounding_box, make_detection_mask, make_image

from torchvision import datapoints
from torchvision.transforms.v2.functional import to_image_pil
from torchvision.transforms.v2.utils import has_all, has_any


IMAGE = make_image(color_space="RGB")
BOUNDING_BOX = make_bounding_box(format=datapoints.BoundingBoxFormat.XYXY, spatial_size=IMAGE.spatial_size)
MASK = make_detection_mask(size=IMAGE.spatial_size)


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image, datapoints.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image, datapoints.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.BoundingBoxes, datapoints.Mask), True),
        ((MASK,), (datapoints.Image, datapoints.BoundingBoxes), False),
        ((BOUNDING_BOX,), (datapoints.Image, datapoints.Mask), False),
        ((IMAGE,), (datapoints.BoundingBoxes, datapoints.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask),
            True,
        ),
        ((), (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda obj: isinstance(obj, datapoints.Image),), True),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
        ((IMAGE,), (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2.utils.is_simple_tensor), True),
        (
            (torch.Tensor(IMAGE),),
            (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2.utils.is_simple_tensor),
            True,
        ),
        (
            (to_image_pil(IMAGE),),
            (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2.utils.is_simple_tensor),
            True,
        ),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image, datapoints.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.Image, datapoints.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (datapoints.BoundingBoxes, datapoints.Mask), True),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (datapoints.Image, datapoints.BoundingBoxes), False),
        ((BOUNDING_BOX, MASK), (datapoints.Image, datapoints.Mask), False),
        ((IMAGE, MASK), (datapoints.BoundingBoxes, datapoints.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask), False),
        ((IMAGE, MASK), (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask), False),
        ((IMAGE, BOUNDING_BOX), (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (lambda obj: isinstance(obj, (datapoints.Image, datapoints.BoundingBoxes, datapoints.Mask)),),
            True,
        ),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
