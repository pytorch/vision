import pytest
import torch

from PIL import Image

from torchvision import datapoints
from torchvision.prototype import datapoints as proto_datapoints


@pytest.mark.parametrize(
    ("data", "input_requires_grad", "expected_requires_grad"),
    [
        ([0.0], None, False),
        ([0.0], False, False),
        ([0.0], True, True),
        (torch.tensor([0.0], requires_grad=False), None, False),
        (torch.tensor([0.0], requires_grad=False), False, False),
        (torch.tensor([0.0], requires_grad=False), True, True),
        (torch.tensor([0.0], requires_grad=True), None, True),
        (torch.tensor([0.0], requires_grad=True), False, False),
        (torch.tensor([0.0], requires_grad=True), True, True),
    ],
)
def test_new_requires_grad(data, input_requires_grad, expected_requires_grad):
    datapoint = proto_datapoints.Label(data, requires_grad=input_requires_grad)
    assert datapoint.requires_grad is expected_requires_grad


def test_isinstance():
    assert isinstance(
        proto_datapoints.Label([0, 1, 0], categories=["foo", "bar"]),
        torch.Tensor,
    )


def test_wrapping_no_copy():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    assert label.data_ptr() == tensor.data_ptr()


def test_to_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    label_to = label.to(torch.int32)

    assert type(label_to) is proto_datapoints.Label
    assert label_to.dtype is torch.int32
    assert label_to.categories is label.categories


def test_to_datapoint_reference():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"]).to(torch.int32)

    tensor_to = tensor.to(label)

    assert type(tensor_to) is torch.Tensor
    assert tensor_to.dtype is torch.int32


def test_clone_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    label_clone = label.clone()

    assert type(label_clone) is proto_datapoints.Label
    assert label_clone.data_ptr() != label.data_ptr()
    assert label_clone.categories is label.categories


def test_requires_grad__wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.float32)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    assert not label.requires_grad

    label_requires_grad = label.requires_grad_(True)

    assert type(label_requires_grad) is proto_datapoints.Label
    assert label.requires_grad
    assert label_requires_grad.requires_grad


def test_other_op_no_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    # any operation besides .to() and .clone() will do here
    output = label * 2

    assert type(output) is torch.Tensor


@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.numpy(),
        lambda t: t.tolist(),
        lambda t: t.max(dim=-1),
    ],
)
def test_no_tensor_output_op_no_wrapping(op):
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    output = op(label)

    assert type(output) is not proto_datapoints.Label


def test_inplace_op_no_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    output = label.add_(0)

    assert type(output) is torch.Tensor
    assert type(label) is proto_datapoints.Label


def test_wrap_like():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = proto_datapoints.Label(tensor, categories=["foo", "bar"])

    # any operation besides .to() and .clone() will do here
    output = label * 2

    label_new = proto_datapoints.Label.wrap_like(label, output)

    assert type(label_new) is proto_datapoints.Label
    assert label_new.data_ptr() == output.data_ptr()
    assert label_new.categories is label.categories


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
