import PIL.Image
import pytest

import torch

import torchvision.transforms.v2._utils
from common_utils import DEFAULT_SIZE, make_bounding_boxes, make_detection_masks, make_image, make_keypoints

from torchvision import tv_tensors
from torchvision.transforms.v2._utils import has_all, has_any
from torchvision.transforms.v2.functional import to_pil_image


IMAGE = make_image(DEFAULT_SIZE, color_space="RGB")
BOUNDING_BOX = make_bounding_boxes(DEFAULT_SIZE, format=tv_tensors.BoundingBoxFormat.XYXY)
MASK = make_detection_masks(DEFAULT_SIZE)
KEYPOINTS = make_keypoints(DEFAULT_SIZE)


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image, tv_tensors.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image, tv_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.BoundingBoxes, tv_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.KeyPoints,), True),
        ((MASK,), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.KeyPoints), False),
        ((BOUNDING_BOX,), (tv_tensors.Image, tv_tensors.Mask, tv_tensors.KeyPoints), False),
        ((IMAGE,), (tv_tensors.BoundingBoxes, tv_tensors.Mask, tv_tensors.KeyPoints), False),
        ((KEYPOINTS,), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK, KEYPOINTS),
            (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask, tv_tensors.KeyPoints),
            True,
        ),
        ((), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask, tv_tensors.KeyPoints), False),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (lambda obj: isinstance(obj, tv_tensors.Image),), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (lambda _: True,), True),
        ((IMAGE,), (tv_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor), True),
        (
            (torch.Tensor(IMAGE),),
            (tv_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
            True,
        ),
        (
            (to_pil_image(IMAGE),),
            (tv_tensors.Image, PIL.Image.Image, torchvision.transforms.v2._utils.is_pure_tensor),
            True,
        ),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.BoundingBoxes,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image, tv_tensors.BoundingBoxes), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Image, tv_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.BoundingBoxes, tv_tensors.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.Mask, tv_tensors.KeyPoints), True),
        ((IMAGE, BOUNDING_BOX, MASK, KEYPOINTS), (tv_tensors.BoundingBoxes, tv_tensors.KeyPoints), True),
        (
            (IMAGE, BOUNDING_BOX, MASK, KEYPOINTS),
            (tv_tensors.BoundingBoxes, tv_tensors.Mask, tv_tensors.KeyPoints),
            True,
        ),
        (
            (IMAGE, BOUNDING_BOX, MASK, KEYPOINTS),
            (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask, tv_tensors.KeyPoints),
            True,
        ),
        ((BOUNDING_BOX, MASK), (tv_tensors.Image, tv_tensors.BoundingBoxes), False),
        ((BOUNDING_BOX, MASK), (tv_tensors.Image, tv_tensors.Mask), False),
        ((IMAGE, MASK), (tv_tensors.BoundingBoxes, tv_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask), False),
        ((IMAGE, MASK), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask), False),
        ((IMAGE, BOUNDING_BOX), (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (lambda obj: isinstance(obj, (tv_tensors.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask)),),
            True,
        ),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
