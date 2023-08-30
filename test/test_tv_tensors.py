from copy import deepcopy

import pytest
import torch
from common_utils import assert_equal, make_bounding_boxes, make_image, make_segmentation_mask, make_video
from PIL import Image

from torchvision import tv_tensors


@pytest.fixture(autouse=True)
def restore_tensor_return_type():
    # This is for security, as we should already be restoring the default manually in each test anyway
    # (at least at the time of writing...)
    yield
    tv_tensors.set_return_type("Tensor")


@pytest.mark.parametrize("data", [torch.rand(3, 32, 32), Image.new("RGB", (32, 32), color=123)])
def test_image_instance(data):
    image = tv_tensors.Image(data)
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3


@pytest.mark.parametrize("data", [torch.randint(0, 10, size=(1, 32, 32)), Image.new("L", (32, 32), color=2)])
def test_mask_instance(data):
    mask = tv_tensors.Mask(data)
    assert isinstance(mask, torch.Tensor)
    assert mask.ndim == 3 and mask.shape[0] == 1


@pytest.mark.parametrize("data", [torch.randint(0, 32, size=(5, 4)), [[0, 0, 5, 5], [2, 2, 7, 7]], [1, 2, 3, 4]])
@pytest.mark.parametrize(
    "format", ["XYXY", "CXCYWH", tv_tensors.BoundingBoxFormat.XYXY, tv_tensors.BoundingBoxFormat.XYWH]
)
def test_bbox_instance(data, format):
    bboxes = tv_tensors.BoundingBoxes(data, format=format, canvas_size=(32, 32))
    assert isinstance(bboxes, torch.Tensor)
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    if isinstance(format, str):
        format = tv_tensors.BoundingBoxFormat[(format.upper())]
    assert bboxes.format == format


def test_bbox_dim_error():
    data_3d = [[[1, 2, 3, 4]]]
    with pytest.raises(ValueError, match="Expected a 1D or 2D tensor, got 3D"):
        tv_tensors.BoundingBoxes(data_3d, format="XYXY", canvas_size=(32, 32))


@pytest.mark.parametrize(
    ("data", "input_requires_grad", "expected_requires_grad"),
    [
        ([[[0.0, 1.0], [0.0, 1.0]]], None, False),
        ([[[0.0, 1.0], [0.0, 1.0]]], False, False),
        ([[[0.0, 1.0], [0.0, 1.0]]], True, True),
        (torch.rand(3, 16, 16, requires_grad=False), None, False),
        (torch.rand(3, 16, 16, requires_grad=False), False, False),
        (torch.rand(3, 16, 16, requires_grad=False), True, True),
        (torch.rand(3, 16, 16, requires_grad=True), None, True),
        (torch.rand(3, 16, 16, requires_grad=True), False, False),
        (torch.rand(3, 16, 16, requires_grad=True), True, True),
    ],
)
def test_new_requires_grad(data, input_requires_grad, expected_requires_grad):
    tv_tensor = tv_tensors.Image(data, requires_grad=input_requires_grad)
    assert tv_tensor.requires_grad is expected_requires_grad


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
def test_isinstance(make_input):
    assert isinstance(make_input(), torch.Tensor)


def test_wrapping_no_copy():
    tensor = torch.rand(3, 16, 16)
    image = tv_tensors.Image(tensor)

    assert image.data_ptr() == tensor.data_ptr()


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
def test_to_wrapping(make_input):
    dp = make_input()

    dp_to = dp.to(torch.float64)

    assert type(dp_to) is type(dp)
    assert dp_to.dtype is torch.float64


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_to_tv_tensor_reference(make_input, return_type):
    tensor = torch.rand((3, 16, 16), dtype=torch.float64)
    dp = make_input()

    with tv_tensors.set_return_type(return_type):
        tensor_to = tensor.to(dp)

    assert type(tensor_to) is (type(dp) if return_type == "TVTensor" else torch.Tensor)
    assert tensor_to.dtype is dp.dtype
    assert type(tensor) is torch.Tensor


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_clone_wrapping(make_input, return_type):
    dp = make_input()

    with tv_tensors.set_return_type(return_type):
        dp_clone = dp.clone()

    assert type(dp_clone) is type(dp)
    assert dp_clone.data_ptr() != dp.data_ptr()


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_requires_grad__wrapping(make_input, return_type):
    dp = make_input(dtype=torch.float)

    assert not dp.requires_grad

    with tv_tensors.set_return_type(return_type):
        dp_requires_grad = dp.requires_grad_(True)

    assert type(dp_requires_grad) is type(dp)
    assert dp.requires_grad
    assert dp_requires_grad.requires_grad


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_detach_wrapping(make_input, return_type):
    dp = make_input(dtype=torch.float).requires_grad_(True)

    with tv_tensors.set_return_type(return_type):
        dp_detached = dp.detach()

    assert type(dp_detached) is type(dp)


@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_force_subclass_with_metadata(return_type):
    # Sanity checks for the ops in _FORCE_TORCHFUNCTION_SUBCLASS and tv_tensors with metadata
    # Largely the same as above, we additionally check that the metadata is preserved
    format, canvas_size = "XYXY", (32, 32)
    bbox = tv_tensors.BoundingBoxes([[0, 0, 5, 5], [2, 2, 7, 7]], format=format, canvas_size=canvas_size)

    tv_tensors.set_return_type(return_type)
    bbox = bbox.clone()
    if return_type == "TVTensor":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    bbox = bbox.to(torch.float64)
    if return_type == "TVTensor":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    bbox = bbox.detach()
    if return_type == "TVTensor":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)

    assert not bbox.requires_grad
    bbox.requires_grad_(True)
    if return_type == "TVTensor":
        assert bbox.format, bbox.canvas_size == (format, canvas_size)
        assert bbox.requires_grad
    tv_tensors.set_return_type("tensor")


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_other_op_no_wrapping(make_input, return_type):
    dp = make_input()

    with tv_tensors.set_return_type(return_type):
        # any operation besides the ones listed in _FORCE_TORCHFUNCTION_SUBCLASS will do here
        output = dp * 2

    assert type(output) is (type(dp) if return_type == "TVTensor" else torch.Tensor)


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.numpy(),
        lambda t: t.tolist(),
        lambda t: t.max(dim=-1),
    ],
)
def test_no_tensor_output_op_no_wrapping(make_input, op):
    dp = make_input()

    output = op(dp)

    assert type(output) is not type(dp)


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
def test_inplace_op_no_wrapping(make_input, return_type):
    dp = make_input()
    original_type = type(dp)

    with tv_tensors.set_return_type(return_type):
        output = dp.add_(0)

    assert type(output) is (type(dp) if return_type == "TVTensor" else torch.Tensor)
    assert type(dp) is original_type


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
def test_wrap(make_input):
    dp = make_input()

    # any operation besides the ones listed in _FORCE_TORCHFUNCTION_SUBCLASS will do here
    output = dp * 2

    dp_new = tv_tensors.wrap(output, like=dp)

    assert type(dp_new) is type(dp)
    assert dp_new.data_ptr() == output.data_ptr()


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_deepcopy(make_input, requires_grad):
    dp = make_input(dtype=torch.float)

    dp.requires_grad_(requires_grad)

    dp_deepcopied = deepcopy(dp)

    assert dp_deepcopied is not dp
    assert dp_deepcopied.data_ptr() != dp.data_ptr()
    assert_equal(dp_deepcopied, dp)

    assert type(dp_deepcopied) is type(dp)
    assert dp_deepcopied.requires_grad is requires_grad


@pytest.mark.parametrize("make_input", [make_image, make_bounding_boxes, make_segmentation_mask, make_video])
@pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
@pytest.mark.parametrize(
    "op",
    (
        lambda dp: dp + torch.rand(*dp.shape),
        lambda dp: torch.rand(*dp.shape) + dp,
        lambda dp: dp * torch.rand(*dp.shape),
        lambda dp: torch.rand(*dp.shape) * dp,
        lambda dp: dp + 3,
        lambda dp: 3 + dp,
        lambda dp: dp + dp,
        lambda dp: dp.sum(),
        lambda dp: dp.reshape(-1),
        lambda dp: dp.int(),
        lambda dp: torch.stack([dp, dp]),
        lambda dp: torch.chunk(dp, 2)[0],
        lambda dp: torch.unbind(dp)[0],
    ),
)
def test_usual_operations(make_input, return_type, op):

    dp = make_input()
    with tv_tensors.set_return_type(return_type):
        out = op(dp)
    assert type(out) is (type(dp) if return_type == "TVTensor" else torch.Tensor)
    if isinstance(dp, tv_tensors.BoundingBoxes) and return_type == "TVTensor":
        assert hasattr(out, "format")
        assert hasattr(out, "canvas_size")


def test_subclasses():
    img = make_image()
    masks = make_segmentation_mask()

    with pytest.raises(TypeError, match="unsupported operand"):
        img + masks


def test_set_return_type():
    img = make_image()

    assert type(img + 3) is torch.Tensor

    with tv_tensors.set_return_type("TVTensor"):
        assert type(img + 3) is tv_tensors.Image
    assert type(img + 3) is torch.Tensor

    tv_tensors.set_return_type("TVTensor")
    assert type(img + 3) is tv_tensors.Image

    with tv_tensors.set_return_type("tensor"):
        assert type(img + 3) is torch.Tensor
        with tv_tensors.set_return_type("TVTensor"):
            assert type(img + 3) is tv_tensors.Image
            tv_tensors.set_return_type("tensor")
            assert type(img + 3) is torch.Tensor
        assert type(img + 3) is torch.Tensor
    # Exiting a context manager will restore the return type as it was prior to entering it,
    # regardless of whether the "global" tv_tensors.set_return_type() was called within the context manager.
    assert type(img + 3) is tv_tensors.Image

    tv_tensors.set_return_type("tensor")


def test_return_type_input():
    img = make_image()

    # Case-insensitive
    with tv_tensors.set_return_type("tvtensor"):
        assert type(img + 3) is tv_tensors.Image

    with pytest.raises(ValueError, match="return_type must be"):
        tv_tensors.set_return_type("typo")

    tv_tensors.set_return_type("tensor")
