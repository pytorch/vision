import PIL.Image
import pytest

import torch

from test_prototype_transforms_functional import make_bounding_box, make_image, make_segmentation_mask

from torchvision.prototype import features
from torchvision.prototype.transforms._utils import has_all, has_any, is_simple_tensor
from torchvision.prototype.transforms.functional import to_image_pil


IMAGE = make_image(color_space=features.ColorSpace.RGB)
BOUNDING_BOX = make_bounding_box(format=features.BoundingBoxFormat.XYXY, image_size=IMAGE.image_size)
SEGMENTATION_MASK = make_segmentation_mask(size=IMAGE.image_size)


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.BoundingBox,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.SegmentationMask,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.BoundingBox), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.SegmentationMask), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.BoundingBox, features.SegmentationMask), True),
        ((SEGMENTATION_MASK,), (features.Image, features.BoundingBox), False),
        ((BOUNDING_BOX,), (features.Image, features.SegmentationMask), False),
        ((IMAGE,), (features.BoundingBox, features.SegmentationMask), False),
        (
            (IMAGE, BOUNDING_BOX, SEGMENTATION_MASK),
            (features.Image, features.BoundingBox, features.SegmentationMask),
            True,
        ),
        ((), (features.Image, features.BoundingBox, features.SegmentationMask), False),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (lambda obj: isinstance(obj, features.Image),), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (lambda _: True,), True),
        ((IMAGE,), (features.Image, PIL.Image.Image, is_simple_tensor), True),
        ((torch.Tensor(IMAGE),), (features.Image, PIL.Image.Image, is_simple_tensor), True),
        ((to_image_pil(IMAGE),), (features.Image, PIL.Image.Image, is_simple_tensor), True),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.BoundingBox,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.SegmentationMask,), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.BoundingBox), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.SegmentationMask), True),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (features.BoundingBox, features.SegmentationMask), True),
        (
            (IMAGE, BOUNDING_BOX, SEGMENTATION_MASK),
            (features.Image, features.BoundingBox, features.SegmentationMask),
            True,
        ),
        ((BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.BoundingBox), False),
        ((BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.SegmentationMask), False),
        ((IMAGE, SEGMENTATION_MASK), (features.BoundingBox, features.SegmentationMask), False),
        (
            (IMAGE, BOUNDING_BOX, SEGMENTATION_MASK),
            (features.Image, features.BoundingBox, features.SegmentationMask),
            True,
        ),
        ((BOUNDING_BOX, SEGMENTATION_MASK), (features.Image, features.BoundingBox, features.SegmentationMask), False),
        ((IMAGE, SEGMENTATION_MASK), (features.Image, features.BoundingBox, features.SegmentationMask), False),
        ((IMAGE, BOUNDING_BOX), (features.Image, features.BoundingBox, features.SegmentationMask), False),
        (
            (IMAGE, BOUNDING_BOX, SEGMENTATION_MASK),
            (lambda obj: isinstance(obj, (features.Image, features.BoundingBox, features.SegmentationMask)),),
            True,
        ),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, SEGMENTATION_MASK), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
