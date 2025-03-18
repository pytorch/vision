import collections.abc
import re

import PIL.Image
import pytest
import torch

from common_utils import assert_equal, make_bounding_boxes, make_detection_masks, make_image, make_video

from torchvision.prototype import transforms, tv_tensors
from torchvision.transforms.v2._utils import check_type, is_pure_tensor
from torchvision.transforms.v2.functional import clamp_bounding_boxes, InterpolationMode, pil_to_tensor, to_pil_image

from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video


def _parse_categories(categories):
    if categories is None:
        num_categories = int(torch.randint(1, 11, ()))
    elif isinstance(categories, int):
        num_categories = categories
        categories = [f"category{idx}" for idx in range(num_categories)]
    elif isinstance(categories, collections.abc.Sequence) and all(isinstance(category, str) for category in categories):
        categories = list(categories)
        num_categories = len(categories)
    else:
        raise pytest.UsageError(
            f"`categories` can either be `None` (default), an integer, or a sequence of strings, "
            f"but got '{categories}' instead."
        )
    return categories, num_categories


def make_label(*, extra_dims=(), categories=10, dtype=torch.int64, device="cpu"):
    categories, num_categories = _parse_categories(categories)
    # The idiom `make_tensor(..., dtype=torch.int64).to(dtype)` is intentional to only get integer values,
    # regardless of the requested dtype, e.g. 0 or 0.0 rather than 0 or 0.123
    data = torch.testing.make_tensor(extra_dims, low=0, high=num_categories, dtype=torch.int64, device=device).to(dtype)
    return tv_tensors.Label(data, categories=categories)


class TestSimpleCopyPaste:
    def create_fake_image(self, mocker, image_type):
        if image_type == PIL.Image.Image:
            return PIL.Image.new("RGB", (32, 32), 123)
        return mocker.MagicMock(spec=image_type)

    def test__extract_image_targets_assertion(self, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, Image),
            # labels, bboxes, masks
            mocker.MagicMock(spec=tv_tensors.Label),
            mocker.MagicMock(spec=BoundingBoxes),
            mocker.MagicMock(spec=Mask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=BoundingBoxes),
            mocker.MagicMock(spec=Mask),
        ]

        with pytest.raises(TypeError, match="requires input sample to contain equal sized list of Images"):
            transform._extract_image_targets(flat_sample)

    @pytest.mark.parametrize("image_type", [Image, PIL.Image.Image, torch.Tensor])
    @pytest.mark.parametrize("label_type", [tv_tensors.Label, tv_tensors.OneHotLabel])
    def test__extract_image_targets(self, image_type, label_type, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, image_type),
            self.create_fake_image(mocker, image_type),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=BoundingBoxes),
            mocker.MagicMock(spec=Mask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=BoundingBoxes),
            mocker.MagicMock(spec=Mask),
        ]

        images, targets = transform._extract_image_targets(flat_sample)

        assert len(images) == len(targets) == 2
        if image_type == PIL.Image.Image:
            torch.testing.assert_close(images[0], pil_to_tensor(flat_sample[0]))
            torch.testing.assert_close(images[1], pil_to_tensor(flat_sample[1]))
        else:
            assert images[0] == flat_sample[0]
            assert images[1] == flat_sample[1]

        for target in targets:
            for key, type_ in [
                ("boxes", BoundingBoxes),
                ("masks", Mask),
                ("labels", label_type),
            ]:
                assert key in target
                assert isinstance(target[key], type_)
                assert target[key] in flat_sample

    @pytest.mark.parametrize("label_type", [tv_tensors.Label, tv_tensors.OneHotLabel])
    def test__copy_paste(self, label_type):
        image = 2 * torch.ones(3, 32, 32)
        masks = torch.zeros(2, 32, 32)
        masks[0, 3:9, 2:8] = 1
        masks[1, 20:30, 20:30] = 1
        labels = torch.tensor([1, 2])
        blending = True
        resize_interpolation = InterpolationMode.BILINEAR
        antialias = None
        if label_type == tv_tensors.OneHotLabel:
            labels = torch.nn.functional.one_hot(labels, num_classes=5)
        target = {
            "boxes": BoundingBoxes(
                torch.tensor([[2.0, 3.0, 8.0, 9.0], [20.0, 20.0, 30.0, 30.0]]), format="XYXY", canvas_size=(32, 32)
            ),
            "masks": Mask(masks),
            "labels": label_type(labels),
        }

        paste_image = 10 * torch.ones(3, 32, 32)
        paste_masks = torch.zeros(2, 32, 32)
        paste_masks[0, 13:19, 12:18] = 1
        paste_masks[1, 15:19, 1:8] = 1
        paste_labels = torch.tensor([3, 4])
        if label_type == tv_tensors.OneHotLabel:
            paste_labels = torch.nn.functional.one_hot(paste_labels, num_classes=5)
        paste_target = {
            "boxes": BoundingBoxes(
                torch.tensor([[12.0, 13.0, 19.0, 18.0], [1.0, 15.0, 8.0, 19.0]]), format="XYXY", canvas_size=(32, 32)
            ),
            "masks": Mask(paste_masks),
            "labels": label_type(paste_labels),
        }

        transform = transforms.SimpleCopyPaste()
        random_selection = torch.tensor([0, 1])
        output_image, output_target = transform._copy_paste(
            image, target, paste_image, paste_target, random_selection, blending, resize_interpolation, antialias
        )

        assert output_image.unique().tolist() == [2, 10]
        assert output_target["boxes"].shape == (4, 4)
        torch.testing.assert_close(output_target["boxes"][:2, :], target["boxes"])
        torch.testing.assert_close(output_target["boxes"][2:, :], paste_target["boxes"])

        expected_labels = torch.tensor([1, 2, 3, 4])
        if label_type == tv_tensors.OneHotLabel:
            expected_labels = torch.nn.functional.one_hot(expected_labels, num_classes=5)
        torch.testing.assert_close(output_target["labels"], label_type(expected_labels))

        assert output_target["masks"].shape == (4, 32, 32)
        torch.testing.assert_close(output_target["masks"][:2, :], target["masks"])
        torch.testing.assert_close(output_target["masks"][2:, :], paste_target["masks"])


class TestFixedSizeCrop:
    def test_make_params(self, mocker):
        crop_size = (7, 7)
        batch_shape = (10,)
        canvas_size = (11, 5)

        transform = transforms.FixedSizeCrop(size=crop_size)

        flat_inputs = [
            make_image(size=canvas_size, color_space="RGB"),
            make_bounding_boxes(format=BoundingBoxFormat.XYXY, canvas_size=canvas_size, num_boxes=batch_shape[0]),
        ]
        params = transform.make_params(flat_inputs)

        assert params["needs_crop"]
        assert params["height"] <= crop_size[0]
        assert params["width"] <= crop_size[1]

        assert (
            isinstance(params["is_valid"], torch.Tensor)
            and params["is_valid"].dtype is torch.bool
            and params["is_valid"].shape == batch_shape
        )

        assert params["needs_pad"]
        assert any(pad > 0 for pad in params["padding"])

    def test__transform_culling(self, mocker):
        batch_size = 10
        canvas_size = (10, 10)

        is_valid = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop.make_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=canvas_size[0],
                width=canvas_size[1],
                is_valid=is_valid,
                needs_pad=False,
            ),
        )

        bounding_boxes = make_bounding_boxes(
            format=BoundingBoxFormat.XYXY, canvas_size=canvas_size, num_boxes=batch_size
        )
        masks = make_detection_masks(size=canvas_size, num_masks=batch_size)
        labels = make_label(extra_dims=(batch_size,))

        transform = transforms.FixedSizeCrop((-1, -1))
        mocker.patch("torchvision.prototype.transforms._geometry.has_any", return_value=True)

        output = transform(
            dict(
                bounding_boxes=bounding_boxes,
                masks=masks,
                labels=labels,
            )
        )

        assert_equal(output["bounding_boxes"], bounding_boxes[is_valid])
        assert_equal(output["masks"], masks[is_valid])
        assert_equal(output["labels"], labels[is_valid])

    def test__transform_bounding_boxes_clamping(self, mocker):
        batch_size = 3
        canvas_size = (10, 10)

        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop.make_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=canvas_size[0],
                width=canvas_size[1],
                is_valid=torch.full((batch_size,), fill_value=True),
                needs_pad=False,
            ),
        )

        bounding_boxes = make_bounding_boxes(
            format=BoundingBoxFormat.XYXY, canvas_size=canvas_size, num_boxes=batch_size
        )
        mock = mocker.patch(
            "torchvision.prototype.transforms._geometry.F.clamp_bounding_boxes", wraps=clamp_bounding_boxes
        )

        transform = transforms.FixedSizeCrop((-1, -1))
        mocker.patch("torchvision.prototype.transforms._geometry.has_any", return_value=True)

        transform(bounding_boxes)

        mock.assert_called_once()


class TestLabelToOneHot:
    def test__transform(self):
        categories = ["apple", "pear", "pineapple"]
        labels = tv_tensors.Label(torch.tensor([0, 1, 2, 1]), categories=categories)
        transform = transforms.LabelToOneHot()
        ohe_labels = transform(labels)
        assert isinstance(ohe_labels, tv_tensors.OneHotLabel)
        assert ohe_labels.shape == (4, 3)
        assert ohe_labels.categories == labels.categories == categories


class TestPermuteDimensions:
    @pytest.mark.parametrize(
        ("dims", "inverse_dims"),
        [
            (
                {Image: (2, 1, 0), Video: None},
                {Image: (2, 1, 0), Video: None},
            ),
            (
                {Image: (2, 1, 0), Video: (1, 2, 3, 0)},
                {Image: (2, 1, 0), Video: (3, 0, 1, 2)},
            ),
        ],
    )
    def test_call(self, dims, inverse_dims):
        sample = dict(
            image=make_image(),
            bounding_boxes=make_bounding_boxes(format=BoundingBoxFormat.XYXY),
            video=make_video(),
            str="str",
            int=0,
        )

        transform = transforms.PermuteDimensions(dims)
        transformed_sample = transform(sample)

        for key, value in sample.items():
            value_type = type(value)
            transformed_value = transformed_sample[key]

            if check_type(value, (Image, is_pure_tensor, Video)):
                if transform.dims.get(value_type) is not None:
                    assert transformed_value.permute(inverse_dims[value_type]).equal(value)
                assert type(transformed_value) == torch.Tensor
            else:
                assert transformed_value is value

    @pytest.mark.filterwarnings("error")
    def test_plain_tensor_call(self):
        tensor = torch.empty((2, 3, 4))
        transform = transforms.PermuteDimensions(dims=(1, 2, 0))

        assert transform(tensor).shape == (3, 4, 2)

    @pytest.mark.parametrize("other_type", [Image, Video])
    def test_plain_tensor_warning(self, other_type):
        with pytest.warns(UserWarning, match=re.escape("`torch.Tensor` will *not* be transformed")):
            transforms.PermuteDimensions(dims={torch.Tensor: (0, 1), other_type: (1, 0)})


class TestTransposeDimensions:
    @pytest.mark.parametrize(
        "dims",
        [
            (-1, -2),
            {Image: (1, 2), Video: None},
        ],
    )
    def test_call(self, dims):
        sample = dict(
            image=make_image(),
            bounding_boxes=make_bounding_boxes(format=BoundingBoxFormat.XYXY),
            video=make_video(),
            str="str",
            int=0,
        )

        transform = transforms.TransposeDimensions(dims)
        transformed_sample = transform(sample)

        for key, value in sample.items():
            value_type = type(value)
            transformed_value = transformed_sample[key]

            transposed_dims = transform.dims.get(value_type)
            if check_type(value, (Image, is_pure_tensor, Video)):
                if transposed_dims is not None:
                    assert transformed_value.transpose(*transposed_dims).equal(value)
                assert type(transformed_value) == torch.Tensor
            else:
                assert transformed_value is value

    @pytest.mark.filterwarnings("error")
    def test_plain_tensor_call(self):
        tensor = torch.empty((2, 3, 4))
        transform = transforms.TransposeDimensions(dims=(0, 2))

        assert transform(tensor).shape == (4, 3, 2)

    @pytest.mark.parametrize("other_type", [Image, Video])
    def test_plain_tensor_warning(self, other_type):
        with pytest.warns(UserWarning, match=re.escape("`torch.Tensor` will *not* be transformed")):
            transforms.TransposeDimensions(dims={torch.Tensor: (0, 1), other_type: (1, 0)})


import importlib.machinery
import importlib.util
from pathlib import Path


def import_transforms_from_references(reference):
    HERE = Path(__file__).parent
    PROJECT_ROOT = HERE.parent

    loader = importlib.machinery.SourceFileLoader(
        "transforms", str(PROJECT_ROOT / "references" / reference / "transforms.py")
    )
    spec = importlib.util.spec_from_loader("transforms", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


det_transforms = import_transforms_from_references("detection")


def test_fixed_sized_crop_against_detection_reference():
    def make_tv_tensors():
        size = (600, 800)
        num_objects = 22

        pil_image = to_pil_image(make_image(size=size, color_space="RGB"))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", num_boxes=num_objects, dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
            "masks": make_detection_masks(size=size, num_masks=num_objects, dtype=torch.long),
        }

        yield (pil_image, target)

        tensor_image = torch.Tensor(make_image(size=size, color_space="RGB"))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", num_boxes=num_objects, dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
            "masks": make_detection_masks(size=size, num_masks=num_objects, dtype=torch.long),
        }

        yield (tensor_image, target)

        tv_tensor_image = make_image(size=size, color_space="RGB")
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", num_boxes=num_objects, dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
            "masks": make_detection_masks(size=size, num_masks=num_objects, dtype=torch.long),
        }

        yield (tv_tensor_image, target)

    t = transforms.FixedSizeCrop((1024, 1024), fill=0)
    t_ref = det_transforms.FixedSizeCrop((1024, 1024), fill=0)

    for dp in make_tv_tensors():
        # We should use prototype transform first as reference transform performs inplace target update
        torch.manual_seed(12)
        output = t(dp)

        torch.manual_seed(12)
        expected_output = t_ref(*dp)

        assert_equal(expected_output, output)
