import re

import pytest
import torch
from PIL import Image

from torchvision import datapoints, datasets


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
        format = datapoints.BoundingBoxFormat.from_str(format.upper())
    assert bboxes.format == format


class TestDatasetWrapper:
    def test_unknown_type(self):
        unknown_object = object()
        with pytest.raises(
            TypeError, match=re.escape("is meant for subclasses of `torchvision.datasets.VisionDataset`")
        ):
            datapoints.wrap_dataset_for_transforms_v2(unknown_object)

    def test_unknown_dataset(self):
        class MyVisionDataset(datasets.VisionDataset):
            pass

        dataset = MyVisionDataset("root")

        with pytest.raises(TypeError, match="No wrapper exist"):
            datapoints.wrap_dataset_for_transforms_v2(dataset)

    def test_missing_wrapper(self):
        dataset = datasets.FakeData()

        with pytest.raises(TypeError, match="please open an issue"):
            datapoints.wrap_dataset_for_transforms_v2(dataset)

    def test_subclass(self, mocker):
        sentinel = object()
        mocker.patch.dict(
            datapoints._dataset_wrapper.WRAPPER_FACTORIES,
            clear=False,
            values={datasets.FakeData: lambda dataset: lambda idx, sample: sentinel},
        )

        class MyFakeData(datasets.FakeData):
            pass

        dataset = MyFakeData()
        wrapped_dataset = datapoints.wrap_dataset_for_transforms_v2(dataset)

        assert wrapped_dataset[0] is sentinel
