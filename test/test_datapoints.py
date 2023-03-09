import pytest
import torch
from PIL import Image

from torchvision import datapoints


@pytest.mark.parametrize("data", [torch.rand(3, 32, 32), Image.new("RGB", (32, 32), color=123)])
def test_image_instance(data):
    image = datapoints.Image(data)
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3


@pytest.mark.parametrize("data", [torch.randint(0, 10, size=(1, 32, 32)), Image.new("L", (32, 32), color=2)])
def test_mask_instance(data):
    mask = datapoints.Mask(data)
    assert isinstance(mask, torch.Tensor)
    assert mask.ndim == 3 and mask.shape[0] == 1


@pytest.mark.parametrize("data", [torch.randint(0, 32, size=(5, 4)), [[0, 0, 5, 5], [2, 2, 7, 7]]])
@pytest.mark.parametrize(
    "format", ["XYXY", "CXCYWH", datapoints.BoundingBoxFormat.XYXY, datapoints.BoundingBoxFormat.XYWH]
)
def test_bbox_instance(data, format):
    bboxes = datapoints.BoundingBox(data, format=format, spatial_size=(32, 32))
    assert isinstance(bboxes, torch.Tensor)
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    if isinstance(format, str):
        format = datapoints.BoundingBoxFormat[(format.upper())]
    assert bboxes.format == format
