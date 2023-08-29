import PIL.Image
import pytest

import torch

import torchvision.transforms.v2._utils
from common_utils import DEFAULT_SIZE, make_bounding_boxes, make_detection_mask, make_image

from torchvision import vision_tensors
from torchvision.transforms.v2._utils import has_all, has_any
from torchvision.transforms.v2.functional import to_pil_image


IMAGE = make_image(DEFAULT_SIZE, color_space="RGB")
BOUNDING_BOX = make_bounding_boxes(DEFAULT_SIZE, format=vision_tensors.BoundingBoxFormat.XYXY)
MASK = make_detection_mask(DEFAULT_SIZE)


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.BoundingBoxes, vision_tensors.Mask), True),
        ((MASK,), (vision_tensors.Image, vision_tensors.BoundingBoxes), False),
        ((BOUNDING_BOX,), (vision_tensors.Image, vision_tensors.Mask), False),
        ((IMAGE,), (vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask),
            True,
        ),
        ((), (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda obj: isinstance(obj, vision_tensors.Image),), True),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
        ((IMAGE,), (vision_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor), True),
        (
            (torch.Tensor(IMAGE),),
            (vision_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
            True,
        ),
        (
            (to_pil_image(IMAGE),),
            (vision_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
            True,
        ),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (vision_tensors.BoundingBoxes, vision_tensors.Mask), True),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.BoundingBoxes), False),
        ((BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.Mask), False),
        ((IMAGE, MASK), (vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        ((IMAGE, MASK), (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        ((IMAGE, BOUNDING_BOX), (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (lambda obj: isinstance(obj, (vision_tensors.Image, vision_tensors.BoundingBoxes, vision_tensors.Mask)),),
            True,
        ),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
