import PIL.Image
import pytest

import torch

import torchvision.transforms.v2._utils
from common_utils import DEFAULT_SIZE, make_bounding_boxes, make_detection_mask, make_image

from torchvision import datapoints
from torchvision.transforms.v2._utils import has_all, has_any
from torchvision.transforms.v2.functional import to_pil_image


IMAGE = make_image(DEFAULT_SIZE, color_space="RGB")
BOUNDING_BOX = make_bounding_boxes(DEFAULT_SIZE, format=datapoints.BoundingBoxFormat.XYXY)
MASK = make_detection_mask(DEFAULT_SIZE)


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
        ((IMAGE,), (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor), True),
        (
            (torch.Tensor(IMAGE),),
            (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
            True,
        ),
        (
            (to_pil_image(IMAGE),),
            (datapoints.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
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
