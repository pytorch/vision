import PIL.Image
import pytest

import torch

from prototype_common_utils import make_bounding_box, make_detection_mask, make_image

from torchvision.prototype import features
from torchvision.prototype.transforms._utils import has_all, has_any
from torchvision.prototype.transforms.functional import to_image_pil


IMAGE = make_image(color_space=features.ColorSpace.RGB)
BOUNDING_BOX = make_bounding_box(format=features.BoundingBoxFormat.XYXY, spatial_size=IMAGE.spatial_size)
MASK = make_detection_mask(size=IMAGE.spatial_size)


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.BoundingBox,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image, features.BoundingBox), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image, features.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.BoundingBox, features.Mask), True),
        ((MASK,), (features.Image, features.BoundingBox), False),
        ((BOUNDING_BOX,), (features.Image, features.Mask), False),
        ((IMAGE,), (features.BoundingBox, features.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (features.Image, features.BoundingBox, features.Mask),
            True,
        ),
        ((), (features.Image, features.BoundingBox, features.Mask), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda obj: isinstance(obj, features.Image),), True),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
        ((IMAGE,), (features.Image, PIL.Image.Image, features.is_simple_tensor), True),
        ((torch.Tensor(IMAGE),), (features.Image, PIL.Image.Image, features.is_simple_tensor), True),
        ((to_image_pil(IMAGE),), (features.Image, PIL.Image.Image, features.is_simple_tensor), True),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.BoundingBox,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Mask,), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image, features.BoundingBox), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.Image, features.Mask), True),
        ((IMAGE, BOUNDING_BOX, MASK), (features.BoundingBox, features.Mask), True),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (features.Image, features.BoundingBox, features.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (features.Image, features.BoundingBox), False),
        ((BOUNDING_BOX, MASK), (features.Image, features.Mask), False),
        ((IMAGE, MASK), (features.BoundingBox, features.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (features.Image, features.BoundingBox, features.Mask),
            True,
        ),
        ((BOUNDING_BOX, MASK), (features.Image, features.BoundingBox, features.Mask), False),
        ((IMAGE, MASK), (features.Image, features.BoundingBox, features.Mask), False),
        ((IMAGE, BOUNDING_BOX), (features.Image, features.BoundingBox, features.Mask), False),
        (
            (IMAGE, BOUNDING_BOX, MASK),
            (lambda obj: isinstance(obj, (features.Image, features.BoundingBox, features.Mask)),),
            True,
        ),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: False,), False),
        ((IMAGE, BOUNDING_BOX, MASK), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
