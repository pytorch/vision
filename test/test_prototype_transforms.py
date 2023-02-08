import itertools
import re

import numpy as np

import PIL.Image
import pytest
import torch

import torchvision.prototype.transforms.utils
from common_utils import cpu_and_gpu
from prototype_common_utils import (
    assert_equal,
    DEFAULT_EXTRA_DIMS,
    make_bounding_box,
    make_bounding_boxes,
    make_detection_mask,
    make_image,
    make_images,
    make_label,
    make_masks,
    make_one_hot_labels,
    make_segmentation_mask,
    make_video,
    make_videos,
)
from torchvision.ops.boxes import box_iou
from torchvision.prototype import datapoints, transforms
from torchvision.prototype.transforms.utils import check_type, is_simple_tensor
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor, to_pil_image

BATCH_EXTRA_DIMS = [extra_dims for extra_dims in DEFAULT_EXTRA_DIMS if extra_dims]


def make_vanilla_tensor_images(*args, **kwargs):
    for image in make_images(*args, **kwargs):
        if image.ndim > 3:
            continue
        yield image.data


def make_pil_images(*args, **kwargs):
    for image in make_vanilla_tensor_images(*args, **kwargs):
        yield to_pil_image(image)


def make_vanilla_tensor_bounding_boxes(*args, **kwargs):
    for bounding_box in make_bounding_boxes(*args, **kwargs):
        yield bounding_box.data


def parametrize(transforms_with_inputs):
    return pytest.mark.parametrize(
        ("transform", "input"),
        [
            pytest.param(
                transform,
                input,
                id=f"{type(transform).__name__}-{type(input).__module__}.{type(input).__name__}-{idx}",
            )
            for transform, inputs in transforms_with_inputs
            for idx, input in enumerate(inputs)
        ],
    )


def parametrize_from_transforms(*transforms):
    transforms_with_inputs = []
    for transform in transforms:
        for creation_fn in [
            make_images,
            make_bounding_boxes,
            make_one_hot_labels,
            make_vanilla_tensor_images,
            make_pil_images,
            make_masks,
            make_videos,
        ]:
            inputs = list(creation_fn())
            try:
                output = transform(inputs[0])
            except Exception:
                continue
            else:
                if output is inputs[0]:
                    continue

            transforms_with_inputs.append((transform, inputs))

    return parametrize(transforms_with_inputs)


class TestSmoke:
    @parametrize_from_transforms(
        transforms.RandomErasing(p=1.0),
        transforms.Resize([16, 16]),
        transforms.CenterCrop([16, 16]),
        transforms.ConvertDtype(),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(5),
        transforms.RandomZoomOut(),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomAffine(degrees=(-45, 45)),
        transforms.RandomCrop([16, 16], padding=1, pad_if_needed=True),
        # TODO: Something wrong with input data setup. Let's fix that
        # transforms.RandomEqualize(),
        # transforms.RandomInvert(),
        # transforms.RandomPosterize(bits=4),
        # transforms.RandomSolarize(threshold=0.5),
        # transforms.RandomAdjustSharpness(sharpness_factor=0.5),
    )
    def test_common(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transform,
                [
                    dict(inpt=inpt, one_hot_label=one_hot_label)
                    for inpt, one_hot_label in itertools.product(
                        itertools.chain(
                            make_images(extra_dims=BATCH_EXTRA_DIMS, dtypes=[torch.float]),
                            make_videos(extra_dims=BATCH_EXTRA_DIMS, dtypes=[torch.float]),
                        ),
                        make_one_hot_labels(extra_dims=BATCH_EXTRA_DIMS, dtypes=[torch.float]),
                    )
                ],
            )
            for transform in [
                transforms.RandomMixup(alpha=1.0),
                transforms.RandomCutmix(alpha=1.0),
            ]
        ]
    )
    def test_mixup_cutmix(self, transform, input):
        transform(input)

        # add other data that should bypass and won't raise any error
        input_copy = dict(input)
        input_copy["path"] = "/path/to/somewhere"
        input_copy["num"] = 1234
        transform(input_copy)

        # Check if we raise an error if sample contains bbox or mask or label
        err_msg = "does not support PIL images, bounding boxes, masks and plain labels"
        input_copy = dict(input)
        for unsup_data in [
            make_label(),
            make_bounding_box(format="XYXY"),
            make_detection_mask(),
            make_segmentation_mask(),
        ]:
            input_copy["unsupported"] = unsup_data
            with pytest.raises(TypeError, match=err_msg):
                transform(input_copy)

    @parametrize(
        [
            (
                transform,
                itertools.chain.from_iterable(
                    fn(
                        color_spaces=[
                            "GRAY",
                            "RGB",
                        ],
                        dtypes=[torch.uint8],
                        extra_dims=[(), (4,)],
                        **(dict(num_frames=["random"]) if fn is make_videos else dict()),
                    )
                    for fn in [
                        make_images,
                        make_vanilla_tensor_images,
                        make_pil_images,
                        make_videos,
                    ]
                ),
            )
            for transform in (
                transforms.RandAugment(),
                transforms.TrivialAugmentWide(),
                transforms.AutoAugment(),
                transforms.AugMix(),
            )
        ]
    )
    def test_auto_augment(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                itertools.chain.from_iterable(
                    fn(color_spaces=["RGB"], dtypes=[torch.float32])
                    for fn in [
                        make_images,
                        make_vanilla_tensor_images,
                        make_videos,
                    ]
                ),
            ),
        ]
    )
    def test_normalize(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transforms.RandomResizedCrop([16, 16]),
                itertools.chain(
                    make_images(extra_dims=[(4,)]),
                    make_vanilla_tensor_images(),
                    make_pil_images(),
                    make_videos(extra_dims=[()]),
                ),
            )
        ]
    )
    def test_random_resized_crop(self, transform, input):
        transform(input)


@pytest.mark.parametrize(
    "flat_inputs",
    itertools.permutations(
        [
            next(make_vanilla_tensor_images()),
            next(make_vanilla_tensor_images()),
            next(make_pil_images()),
            make_image(),
            next(make_videos()),
        ],
        3,
    ),
)
def test_simple_tensor_heuristic(flat_inputs):
    def split_on_simple_tensor(to_split):
        # This takes a sequence that is structurally aligned with `flat_inputs` and splits its items into three parts:
        # 1. The first simple tensor. If none is present, this will be `None`
        # 2. A list of the remaining simple tensors
        # 3. A list of all other items
        simple_tensors = []
        others = []
        # Splitting always happens on the original `flat_inputs` to avoid any erroneous type changes by the transform to
        # affect the splitting.
        for item, inpt in zip(to_split, flat_inputs):
            (simple_tensors if is_simple_tensor(inpt) else others).append(item)
        return simple_tensors[0] if simple_tensors else None, simple_tensors[1:], others

    class CopyCloneTransform(transforms.Transform):
        def _transform(self, inpt, params):
            return inpt.clone() if isinstance(inpt, torch.Tensor) else inpt.copy()

        @staticmethod
        def was_applied(output, inpt):
            identity = output is inpt
            if identity:
                return False

            # Make sure nothing fishy is going on
            assert_equal(output, inpt)
            return True

    first_simple_tensor_input, other_simple_tensor_inputs, other_inputs = split_on_simple_tensor(flat_inputs)

    transform = CopyCloneTransform()
    transformed_sample = transform(flat_inputs)

    first_simple_tensor_output, other_simple_tensor_outputs, other_outputs = split_on_simple_tensor(transformed_sample)

    if first_simple_tensor_input is not None:
        if other_inputs:
            assert not transform.was_applied(first_simple_tensor_output, first_simple_tensor_input)
        else:
            assert transform.was_applied(first_simple_tensor_output, first_simple_tensor_input)

    for output, inpt in zip(other_simple_tensor_outputs, other_simple_tensor_inputs):
        assert not transform.was_applied(output, inpt)

    for input, output in zip(other_inputs, other_outputs):
        assert transform.was_applied(output, input)


@pytest.mark.parametrize("p", [0.0, 1.0])
class TestRandomHorizontalFlip:
    def input_expected_image_tensor(self, p, dtype=torch.float32):
        input = torch.tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], dtype=dtype)
        expected = torch.tensor([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=dtype)

        return input, expected if p == 1 else input

    def test_simple_tensor(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(input)

        assert_equal(expected, actual)

    def test_pil_image(self, p):
        input, expected = self.input_expected_image_tensor(p, dtype=torch.uint8)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(to_pil_image(input))

        assert_equal(expected, pil_to_tensor(actual))

    def test_datapoints_image(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(datapoints.Image(input))

        assert_equal(datapoints.Image(expected), actual)

    def test_datapoints_mask(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(datapoints.Mask(input))

        assert_equal(datapoints.Mask(expected), actual)

    def test_datapoints_bounding_box(self, p):
        input = datapoints.BoundingBox([0, 0, 5, 5], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(10, 10))
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(input)

        expected_image_tensor = torch.tensor([5, 0, 10, 5]) if p == 1.0 else input
        expected = datapoints.BoundingBox.wrap_like(input, expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.spatial_size == expected.spatial_size


@pytest.mark.parametrize("p", [0.0, 1.0])
class TestRandomVerticalFlip:
    def input_expected_image_tensor(self, p, dtype=torch.float32):
        input = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=dtype)
        expected = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=dtype)

        return input, expected if p == 1 else input

    def test_simple_tensor(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(input)

        assert_equal(expected, actual)

    def test_pil_image(self, p):
        input, expected = self.input_expected_image_tensor(p, dtype=torch.uint8)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(to_pil_image(input))

        assert_equal(expected, pil_to_tensor(actual))

    def test_datapoints_image(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(datapoints.Image(input))

        assert_equal(datapoints.Image(expected), actual)

    def test_datapoints_mask(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(datapoints.Mask(input))

        assert_equal(datapoints.Mask(expected), actual)

    def test_datapoints_bounding_box(self, p):
        input = datapoints.BoundingBox([0, 0, 5, 5], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(10, 10))
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(input)

        expected_image_tensor = torch.tensor([0, 5, 5, 10]) if p == 1.0 else input
        expected = datapoints.BoundingBox.wrap_like(input, expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.spatial_size == expected.spatial_size


class TestPad:
    def test_assertions(self):
        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.Pad("abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.Pad([-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.Pad(12, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.Pad(12, padding_mode="abc")

    @pytest.mark.parametrize("padding", [1, (1, 2), [1, 2, 3, 4]])
    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("padding_mode", ["constant", "edge"])
    def test__transform(self, padding, fill, padding_mode, mocker):
        transform = transforms.Pad(padding, fill=fill, padding_mode=padding_mode)

        fn = mocker.patch("torchvision.prototype.transforms.functional.pad")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        _ = transform(inpt)

        fill = transforms._utils._convert_fill_arg(fill)
        if isinstance(padding, tuple):
            padding = list(padding)
        fn.assert_called_once_with(inpt, padding=padding, fill=fill, padding_mode=padding_mode)

    @pytest.mark.parametrize("fill", [12, {datapoints.Image: 12, datapoints.Mask: 34}])
    def test__transform_image_mask(self, fill, mocker):
        transform = transforms.Pad(1, fill=fill, padding_mode="constant")

        fn = mocker.patch("torchvision.prototype.transforms.functional.pad")
        image = datapoints.Image(torch.rand(3, 32, 32))
        mask = datapoints.Mask(torch.randint(0, 5, size=(32, 32)))
        inpt = [image, mask]
        _ = transform(inpt)

        if isinstance(fill, int):
            fill = transforms._utils._convert_fill_arg(fill)
            calls = [
                mocker.call(image, padding=1, fill=fill, padding_mode="constant"),
                mocker.call(mask, padding=1, fill=fill, padding_mode="constant"),
            ]
        else:
            fill_img = transforms._utils._convert_fill_arg(fill[type(image)])
            fill_mask = transforms._utils._convert_fill_arg(fill[type(mask)])
            calls = [
                mocker.call(image, padding=1, fill=fill_img, padding_mode="constant"),
                mocker.call(mask, padding=1, fill=fill_mask, padding_mode="constant"),
            ]
        fn.assert_has_calls(calls)


class TestRandomZoomOut:
    def test_assertions(self):
        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomZoomOut(fill="abc")

        with pytest.raises(TypeError, match="should be a sequence of length"):
            transforms.RandomZoomOut(0, side_range=0)

        with pytest.raises(ValueError, match="Invalid canvas side range"):
            transforms.RandomZoomOut(0, side_range=[4.0, 1.0])

    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    def test__get_params(self, fill, side_range, mocker):
        transform = transforms.RandomZoomOut(fill=fill, side_range=side_range)

        image = mocker.MagicMock(spec=datapoints.Image)
        h, w = image.spatial_size = (24, 32)

        params = transform._get_params([image])

        assert len(params["padding"]) == 4
        assert 0 <= params["padding"][0] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][1] <= (side_range[1] - 1) * h
        assert 0 <= params["padding"][2] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][3] <= (side_range[1] - 1) * h

    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    def test__transform(self, fill, side_range, mocker):
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (24, 32)

        transform = transforms.RandomZoomOut(fill=fill, side_range=side_range, p=1)

        fn = mocker.patch("torchvision.prototype.transforms.functional.pad")
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        torch.rand(1)  # random apply changes random state
        params = transform._get_params([inpt])

        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, **params, fill=fill)

    @pytest.mark.parametrize("fill", [12, {datapoints.Image: 12, datapoints.Mask: 34}])
    def test__transform_image_mask(self, fill, mocker):
        transform = transforms.RandomZoomOut(fill=fill, p=1.0)

        fn = mocker.patch("torchvision.prototype.transforms.functional.pad")
        image = datapoints.Image(torch.rand(3, 32, 32))
        mask = datapoints.Mask(torch.randint(0, 5, size=(32, 32)))
        inpt = [image, mask]

        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        torch.rand(1)  # random apply changes random state
        params = transform._get_params(inpt)

        if isinstance(fill, int):
            fill = transforms._utils._convert_fill_arg(fill)
            calls = [
                mocker.call(image, **params, fill=fill),
                mocker.call(mask, **params, fill=fill),
            ]
        else:
            fill_img = transforms._utils._convert_fill_arg(fill[type(image)])
            fill_mask = transforms._utils._convert_fill_arg(fill[type(mask)])
            calls = [
                mocker.call(image, **params, fill=fill_img),
                mocker.call(mask, **params, fill=fill_mask),
            ]
        fn.assert_has_calls(calls)


class TestRandomRotation:
    def test_assertions(self):
        with pytest.raises(ValueError, match="is a single number, it must be positive"):
            transforms.RandomRotation(-0.7)

        for d in [[-0.7], [-0.7, 0, 0.7]]:
            with pytest.raises(ValueError, match="degrees should be a sequence of length 2"):
                transforms.RandomRotation(d)

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomRotation(12, fill="abc")

        with pytest.raises(TypeError, match="center should be a sequence of length"):
            transforms.RandomRotation(12, center=12)

        with pytest.raises(ValueError, match="center should be a sequence of length"):
            transforms.RandomRotation(12, center=[1, 2, 3])

    def test__get_params(self):
        angle_bound = 34
        transform = transforms.RandomRotation(angle_bound)

        params = transform._get_params(None)
        assert -angle_bound <= params["angle"] <= angle_bound

        angle_bounds = [12, 34]
        transform = transforms.RandomRotation(angle_bounds)

        params = transform._get_params(None)
        assert angle_bounds[0] <= params["angle"] <= angle_bounds[1]

    @pytest.mark.parametrize("degrees", [23, [0, 45], (0, 45)])
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("center", [None, [2.0, 3.0]])
    def test__transform(self, degrees, expand, fill, center, mocker):
        interpolation = InterpolationMode.BILINEAR
        transform = transforms.RandomRotation(
            degrees, interpolation=interpolation, expand=expand, fill=fill, center=center
        )

        if isinstance(degrees, (tuple, list)):
            assert transform.degrees == [float(degrees[0]), float(degrees[1])]
        else:
            assert transform.degrees == [float(-degrees), float(degrees)]

        fn = mocker.patch("torchvision.prototype.transforms.functional.rotate")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params(inpt)

        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, **params, interpolation=interpolation, expand=expand, fill=fill, center=center)

    @pytest.mark.parametrize("angle", [34, -87])
    @pytest.mark.parametrize("expand", [False, True])
    def test_boundingbox_spatial_size(self, angle, expand):
        # Specific test for BoundingBox.rotate
        bbox = datapoints.BoundingBox(
            torch.tensor([1, 2, 3, 4]), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=(32, 32)
        )
        img = datapoints.Image(torch.rand(1, 3, 32, 32))

        out_img = img.rotate(angle, expand=expand)
        out_bbox = bbox.rotate(angle, expand=expand)

        assert out_img.spatial_size == out_bbox.spatial_size


class TestRandomAffine:
    def test_assertions(self):
        with pytest.raises(ValueError, match="is a single number, it must be positive"):
            transforms.RandomAffine(-0.7)

        for d in [[-0.7], [-0.7, 0, 0.7]]:
            with pytest.raises(ValueError, match="degrees should be a sequence of length 2"):
                transforms.RandomAffine(d)

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(12, fill="abc")

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(12, fill="abc")

        for kwargs in [
            {"center": 12},
            {"translate": 12},
            {"scale": 12},
        ]:
            with pytest.raises(TypeError, match="should be a sequence of length"):
                transforms.RandomAffine(12, **kwargs)

        for kwargs in [{"center": [1, 2, 3]}, {"translate": [1, 2, 3]}, {"scale": [1, 2, 3]}]:
            with pytest.raises(ValueError, match="should be a sequence of length"):
                transforms.RandomAffine(12, **kwargs)

        with pytest.raises(ValueError, match="translation values should be between 0 and 1"):
            transforms.RandomAffine(12, translate=[-1.0, 2.0])

        with pytest.raises(ValueError, match="scale values should be positive"):
            transforms.RandomAffine(12, scale=[-1.0, 2.0])

        with pytest.raises(ValueError, match="is a single number, it must be positive"):
            transforms.RandomAffine(12, shear=-10)

        for s in [[-0.7], [-0.7, 0, 0.7]]:
            with pytest.raises(ValueError, match="shear should be a sequence of length 2"):
                transforms.RandomAffine(12, shear=s)

    @pytest.mark.parametrize("degrees", [23, [0, 45], (0, 45)])
    @pytest.mark.parametrize("translate", [None, [0.1, 0.2]])
    @pytest.mark.parametrize("scale", [None, [0.7, 1.2]])
    @pytest.mark.parametrize("shear", [None, 2.0, [5.0, 15.0], [1.0, 2.0, 3.0, 4.0]])
    def test__get_params(self, degrees, translate, scale, shear, mocker):
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)
        h, w = image.spatial_size

        transform = transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear)
        params = transform._get_params([image])

        if not isinstance(degrees, (list, tuple)):
            assert -degrees <= params["angle"] <= degrees
        else:
            assert degrees[0] <= params["angle"] <= degrees[1]

        if translate is not None:
            w_max = int(round(translate[0] * w))
            h_max = int(round(translate[1] * h))
            assert -w_max <= params["translate"][0] <= w_max
            assert -h_max <= params["translate"][1] <= h_max
        else:
            assert params["translate"] == (0, 0)

        if scale is not None:
            assert scale[0] <= params["scale"] <= scale[1]
        else:
            assert params["scale"] == 1.0

        if shear is not None:
            if isinstance(shear, float):
                assert -shear <= params["shear"][0] <= shear
                assert params["shear"][1] == 0.0
            elif len(shear) == 2:
                assert shear[0] <= params["shear"][0] <= shear[1]
                assert params["shear"][1] == 0.0
            else:
                assert shear[0] <= params["shear"][0] <= shear[1]
                assert shear[2] <= params["shear"][1] <= shear[3]
        else:
            assert params["shear"] == (0, 0)

    @pytest.mark.parametrize("degrees", [23, [0, 45], (0, 45)])
    @pytest.mark.parametrize("translate", [None, [0.1, 0.2]])
    @pytest.mark.parametrize("scale", [None, [0.7, 1.2]])
    @pytest.mark.parametrize("shear", [None, 2.0, [5.0, 15.0], [1.0, 2.0, 3.0, 4.0]])
    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("center", [None, [2.0, 3.0]])
    def test__transform(self, degrees, translate, scale, shear, fill, center, mocker):
        interpolation = InterpolationMode.BILINEAR
        transform = transforms.RandomAffine(
            degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )

        if isinstance(degrees, (tuple, list)):
            assert transform.degrees == [float(degrees[0]), float(degrees[1])]
        else:
            assert transform.degrees == [float(-degrees), float(degrees)]

        fn = mocker.patch("torchvision.prototype.transforms.functional.affine")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (24, 32)

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params([inpt])

        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, **params, interpolation=interpolation, fill=fill, center=center)


class TestRandomCrop:
    def test_assertions(self):
        with pytest.raises(ValueError, match="Please provide only two dimensions"):
            transforms.RandomCrop([10, 12, 14])

        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.RandomCrop([10, 12], padding="abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.RandomCrop([10, 12], padding=[-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomCrop([10, 12], padding=1, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.RandomCrop([10, 12], padding=1, padding_mode="abc")

    @pytest.mark.parametrize("padding", [None, 1, [2, 3], [1, 2, 3, 4]])
    @pytest.mark.parametrize("size, pad_if_needed", [((10, 10), False), ((50, 25), True)])
    def test__get_params(self, padding, pad_if_needed, size, mocker):
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)
        h, w = image.spatial_size

        transform = transforms.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed)
        params = transform._get_params([image])

        if padding is not None:
            if isinstance(padding, int):
                pad_top = pad_bottom = pad_left = pad_right = padding
            elif isinstance(padding, list) and len(padding) == 2:
                pad_left = pad_right = padding[0]
                pad_top = pad_bottom = padding[1]
            elif isinstance(padding, list) and len(padding) == 4:
                pad_left, pad_top, pad_right, pad_bottom = padding

            h += pad_top + pad_bottom
            w += pad_left + pad_right
        else:
            pad_left = pad_right = pad_top = pad_bottom = 0

        if pad_if_needed:
            if w < size[1]:
                diff = size[1] - w
                pad_left += diff
                pad_right += diff
                w += 2 * diff
            if h < size[0]:
                diff = size[0] - h
                pad_top += diff
                pad_bottom += diff
                h += 2 * diff

        padding = [pad_left, pad_top, pad_right, pad_bottom]

        assert 0 <= params["top"] <= h - size[0] + 1
        assert 0 <= params["left"] <= w - size[1] + 1
        assert params["height"] == size[0]
        assert params["width"] == size[1]
        assert params["needs_pad"] is any(padding)
        assert params["padding"] == padding

    @pytest.mark.parametrize("padding", [None, 1, [2, 3], [1, 2, 3, 4]])
    @pytest.mark.parametrize("pad_if_needed", [False, True])
    @pytest.mark.parametrize("fill", [False, True])
    @pytest.mark.parametrize("padding_mode", ["constant", "edge"])
    def test__transform(self, padding, pad_if_needed, fill, padding_mode, mocker):
        output_size = [10, 12]
        transform = transforms.RandomCrop(
            output_size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode
        )

        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (32, 32)

        expected = mocker.MagicMock(spec=datapoints.Image)
        expected.num_channels = 3
        if isinstance(padding, int):
            expected.spatial_size = (inpt.spatial_size[0] + padding, inpt.spatial_size[1] + padding)
        elif isinstance(padding, list):
            expected.spatial_size = (
                inpt.spatial_size[0] + sum(padding[0::2]),
                inpt.spatial_size[1] + sum(padding[1::2]),
            )
        else:
            expected.spatial_size = inpt.spatial_size
        _ = mocker.patch("torchvision.prototype.transforms.functional.pad", return_value=expected)
        fn_crop = mocker.patch("torchvision.prototype.transforms.functional.crop")

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params([inpt])
        if padding is None and not pad_if_needed:
            fn_crop.assert_called_once_with(
                inpt, top=params["top"], left=params["left"], height=output_size[0], width=output_size[1]
            )
        elif not pad_if_needed:
            fn_crop.assert_called_once_with(
                expected, top=params["top"], left=params["left"], height=output_size[0], width=output_size[1]
            )
        elif padding is None:
            # vfdev-5: I do not know how to mock and test this case
            pass
        else:
            # vfdev-5: I do not know how to mock and test this case
            pass


class TestGaussianBlur:
    def test_assertions(self):
        with pytest.raises(ValueError, match="Kernel size should be a tuple/list of two integers"):
            transforms.GaussianBlur([10, 12, 14])

        with pytest.raises(ValueError, match="Kernel size value should be an odd and positive number"):
            transforms.GaussianBlur(4)

        with pytest.raises(
            TypeError, match="sigma should be a single int or float or a list/tuple with length 2 floats."
        ):
            transforms.GaussianBlur(3, sigma=[1, 2, 3])

        with pytest.raises(ValueError, match="If sigma is a single number, it must be positive"):
            transforms.GaussianBlur(3, sigma=-1.0)

        with pytest.raises(ValueError, match="sigma values should be positive and of the form"):
            transforms.GaussianBlur(3, sigma=[2.0, 1.0])

    @pytest.mark.parametrize("sigma", [10.0, [10.0, 12.0]])
    def test__get_params(self, sigma):
        transform = transforms.GaussianBlur(3, sigma=sigma)
        params = transform._get_params([])

        if isinstance(sigma, float):
            assert params["sigma"][0] == params["sigma"][1] == 10
        else:
            assert sigma[0] <= params["sigma"][0] <= sigma[1]
            assert sigma[0] <= params["sigma"][1] <= sigma[1]

    @pytest.mark.parametrize("kernel_size", [3, [3, 5], (5, 3)])
    @pytest.mark.parametrize("sigma", [2.0, [2.0, 3.0]])
    def test__transform(self, kernel_size, sigma, mocker):
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

        if isinstance(kernel_size, (tuple, list)):
            assert transform.kernel_size == kernel_size
        else:
            kernel_size = (kernel_size, kernel_size)
            assert transform.kernel_size == kernel_size

        if isinstance(sigma, (tuple, list)):
            assert transform.sigma == sigma
        else:
            assert transform.sigma == [sigma, sigma]

        fn = mocker.patch("torchvision.prototype.transforms.functional.gaussian_blur")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (24, 32)

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params([inpt])

        fn.assert_called_once_with(inpt, kernel_size, **params)


class TestRandomColorOp:
    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize(
        "transform_cls, func_op_name, kwargs",
        [
            (transforms.RandomEqualize, "equalize", {}),
            (transforms.RandomInvert, "invert", {}),
            (transforms.RandomAutocontrast, "autocontrast", {}),
            (transforms.RandomPosterize, "posterize", {"bits": 4}),
            (transforms.RandomSolarize, "solarize", {"threshold": 0.5}),
            (transforms.RandomAdjustSharpness, "adjust_sharpness", {"sharpness_factor": 0.5}),
        ],
    )
    def test__transform(self, p, transform_cls, func_op_name, kwargs, mocker):
        transform = transform_cls(p=p, **kwargs)

        fn = mocker.patch(f"torchvision.prototype.transforms.functional.{func_op_name}")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        _ = transform(inpt)
        if p > 0.0:
            fn.assert_called_once_with(inpt, **kwargs)
        else:
            assert fn.call_count == 0


class TestRandomPerspective:
    def test_assertions(self):
        with pytest.raises(ValueError, match="Argument distortion_scale value should be between 0 and 1"):
            transforms.RandomPerspective(distortion_scale=-1.0)

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomPerspective(0.5, fill="abc")

    def test__get_params(self, mocker):
        dscale = 0.5
        transform = transforms.RandomPerspective(dscale)
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)

        params = transform._get_params([image])

        h, w = image.spatial_size
        assert "coefficients" in params
        assert len(params["coefficients"]) == 8

    @pytest.mark.parametrize("distortion_scale", [0.1, 0.7])
    def test__transform(self, distortion_scale, mocker):
        interpolation = InterpolationMode.BILINEAR
        fill = 12
        transform = transforms.RandomPerspective(distortion_scale, fill=fill, interpolation=interpolation)

        fn = mocker.patch("torchvision.prototype.transforms.functional.perspective")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (24, 32)
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        torch.rand(1)  # random apply changes random state
        params = transform._get_params([inpt])

        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, None, None, **params, fill=fill, interpolation=interpolation)


class TestElasticTransform:
    def test_assertions(self):

        with pytest.raises(TypeError, match="alpha should be float or a sequence of floats"):
            transforms.ElasticTransform({})

        with pytest.raises(ValueError, match="alpha is a sequence its length should be one of 2"):
            transforms.ElasticTransform([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="alpha should be a sequence of floats"):
            transforms.ElasticTransform([1, 2])

        with pytest.raises(TypeError, match="sigma should be float or a sequence of floats"):
            transforms.ElasticTransform(1.0, {})

        with pytest.raises(ValueError, match="sigma is a sequence its length should be one of 2"):
            transforms.ElasticTransform(1.0, [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="sigma should be a sequence of floats"):
            transforms.ElasticTransform(1.0, [1, 2])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.ElasticTransform(1.0, 2.0, fill="abc")

    def test__get_params(self, mocker):
        alpha = 2.0
        sigma = 3.0
        transform = transforms.ElasticTransform(alpha, sigma)
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)

        params = transform._get_params([image])

        h, w = image.spatial_size
        displacement = params["displacement"]
        assert displacement.shape == (1, h, w, 2)
        assert (-alpha / w <= displacement[0, ..., 0]).all() and (displacement[0, ..., 0] <= alpha / w).all()
        assert (-alpha / h <= displacement[0, ..., 1]).all() and (displacement[0, ..., 1] <= alpha / h).all()

    @pytest.mark.parametrize("alpha", [5.0, [5.0, 10.0]])
    @pytest.mark.parametrize("sigma", [2.0, [2.0, 5.0]])
    def test__transform(self, alpha, sigma, mocker):
        interpolation = InterpolationMode.BILINEAR
        fill = 12
        transform = transforms.ElasticTransform(alpha, sigma=sigma, fill=fill, interpolation=interpolation)

        if isinstance(alpha, float):
            assert transform.alpha == [alpha, alpha]
        else:
            assert transform.alpha == alpha

        if isinstance(sigma, float):
            assert transform.sigma == [sigma, sigma]
        else:
            assert transform.sigma == sigma

        fn = mocker.patch("torchvision.prototype.transforms.functional.elastic")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.spatial_size = (24, 32)

        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock()
        _ = transform(inpt)
        params = transform._get_params([inpt])
        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, **params, fill=fill, interpolation=interpolation)


class TestRandomErasing:
    def test_assertions(self, mocker):
        with pytest.raises(TypeError, match="Argument value should be either a number or str or a sequence"):
            transforms.RandomErasing(value={})

        with pytest.raises(ValueError, match="If value is str, it should be 'random'"):
            transforms.RandomErasing(value="abc")

        with pytest.raises(TypeError, match="Scale should be a sequence"):
            transforms.RandomErasing(scale=123)

        with pytest.raises(TypeError, match="Ratio should be a sequence"):
            transforms.RandomErasing(ratio=123)

        with pytest.raises(ValueError, match="Scale should be between 0 and 1"):
            transforms.RandomErasing(scale=[-1, 2])

        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)

        transform = transforms.RandomErasing(value=[1, 2, 3, 4])

        with pytest.raises(ValueError, match="If value is a sequence, it should have either a single value"):
            transform._get_params([image])

    @pytest.mark.parametrize("value", [5.0, [1, 2, 3], "random"])
    def test__get_params(self, value, mocker):
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)

        transform = transforms.RandomErasing(value=value)
        params = transform._get_params([image])

        v = params["v"]
        h, w = params["h"], params["w"]
        i, j = params["i"], params["j"]
        assert isinstance(v, torch.Tensor)
        if value == "random":
            assert v.shape == (image.num_channels, h, w)
        elif isinstance(value, (int, float)):
            assert v.shape == (1, 1, 1)
        elif isinstance(value, (list, tuple)):
            assert v.shape == (image.num_channels, 1, 1)

        assert 0 <= i <= image.spatial_size[0] - h
        assert 0 <= j <= image.spatial_size[1] - w

    @pytest.mark.parametrize("p", [0, 1])
    def test__transform(self, mocker, p):
        transform = transforms.RandomErasing(p=p)
        transform._transformed_types = (mocker.MagicMock,)

        i_sentinel = mocker.MagicMock()
        j_sentinel = mocker.MagicMock()
        h_sentinel = mocker.MagicMock()
        w_sentinel = mocker.MagicMock()
        v_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._augment.RandomErasing._get_params",
            return_value=dict(i=i_sentinel, j=j_sentinel, h=h_sentinel, w=w_sentinel, v=v_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.prototype.transforms._augment.F.erase")
        output = transform(inpt_sentinel)

        if p:
            mock.assert_called_once_with(
                inpt_sentinel,
                i=i_sentinel,
                j=j_sentinel,
                h=h_sentinel,
                w=w_sentinel,
                v=v_sentinel,
                inplace=transform.inplace,
            )
        else:
            mock.assert_not_called()
            assert output is inpt_sentinel


class TestTransform:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBox, str, int],
    )
    def test_check_transformed_types(self, inpt_type, mocker):
        # This test ensures that we correctly handle which types to transform and which to bypass
        t = transforms.Transform()
        inpt = mocker.MagicMock(spec=inpt_type)

        if inpt_type in (np.ndarray, str, int):
            output = t(inpt)
            assert output is inpt
        else:
            with pytest.raises(NotImplementedError):
                t(inpt)


class TestToImageTensor:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch(
            "torchvision.prototype.transforms.functional.to_image_tensor",
            return_value=torch.rand(1, 3, 8, 8),
        )

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImageTensor()
        transform(inpt)
        if inpt_type in (datapoints.BoundingBox, datapoints.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt)


class TestToImagePIL:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.prototype.transforms.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImagePIL()
        transform(inpt)
        if inpt_type in (datapoints.BoundingBox, PIL.Image.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToPILImage:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.prototype.transforms.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToPILImage()
        transform(inpt)
        if inpt_type in (PIL.Image.Image, datapoints.BoundingBox, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToTensor:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.functional.to_tensor")

        inpt = mocker.MagicMock(spec=inpt_type)
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            transform = transforms.ToTensor()
        transform(inpt)
        if inpt_type in (datapoints.Image, torch.Tensor, datapoints.BoundingBox, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt)


class TestContainers:
    @pytest.mark.parametrize("transform_cls", [transforms.Compose, transforms.RandomChoice, transforms.RandomOrder])
    def test_assertions(self, transform_cls):
        with pytest.raises(TypeError, match="Argument transforms should be a sequence of callables"):
            transform_cls(transforms.RandomCrop(28))

    @pytest.mark.parametrize("transform_cls", [transforms.Compose, transforms.RandomChoice, transforms.RandomOrder])
    @pytest.mark.parametrize(
        "trfms",
        [
            [transforms.Pad(2), transforms.RandomCrop(28)],
            [lambda x: 2.0 * x, transforms.Pad(2), transforms.RandomCrop(28)],
            [transforms.Pad(2), lambda x: 2.0 * x, transforms.RandomCrop(28)],
        ],
    )
    def test_ctor(self, transform_cls, trfms):
        c = transform_cls(trfms)
        inpt = torch.rand(1, 3, 32, 32)
        output = c(inpt)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 4


class TestRandomChoice:
    def test_assertions(self):
        with pytest.warns(UserWarning, match="Argument p is deprecated and will be removed"):
            transforms.RandomChoice([transforms.Pad(2), transforms.RandomCrop(28)], p=[1, 2])

        with pytest.raises(ValueError, match="The number of probabilities doesn't match the number of transforms"):
            transforms.RandomChoice([transforms.Pad(2), transforms.RandomCrop(28)], probabilities=[1])


class TestRandomIoUCrop:
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("options", [[0.5, 0.9], [2.0]])
    def test__get_params(self, device, options, mocker):
        image = mocker.MagicMock(spec=datapoints.Image)
        image.num_channels = 3
        image.spatial_size = (24, 32)
        bboxes = datapoints.BoundingBox(
            torch.tensor([[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]),
            format="XYXY",
            spatial_size=image.spatial_size,
            device=device,
        )
        sample = [image, bboxes]

        transform = transforms.RandomIoUCrop(sampler_options=options)

        n_samples = 5
        for _ in range(n_samples):

            params = transform._get_params(sample)

            if options == [2.0]:
                assert len(params) == 0
                return

            assert len(params["is_within_crop_area"]) > 0
            assert params["is_within_crop_area"].dtype == torch.bool

            orig_h = image.spatial_size[0]
            orig_w = image.spatial_size[1]
            assert int(transform.min_scale * orig_h) <= params["height"] <= int(transform.max_scale * orig_h)
            assert int(transform.min_scale * orig_w) <= params["width"] <= int(transform.max_scale * orig_w)

            left, top = params["left"], params["top"]
            new_h, new_w = params["height"], params["width"]
            ious = box_iou(
                bboxes,
                torch.tensor([[left, top, left + new_w, top + new_h]], dtype=bboxes.dtype, device=bboxes.device),
            )
            assert ious.max() >= options[0] or ious.max() >= options[1], f"{ious} vs {options}"

    def test__transform_empty_params(self, mocker):
        transform = transforms.RandomIoUCrop(sampler_options=[2.0])
        image = datapoints.Image(torch.rand(1, 3, 4, 4))
        bboxes = datapoints.BoundingBox(torch.tensor([[1, 1, 2, 2]]), format="XYXY", spatial_size=(4, 4))
        label = datapoints.Label(torch.tensor([1]))
        sample = [image, bboxes, label]
        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock(return_value={})
        output = transform(sample)
        torch.testing.assert_close(output, sample)

    def test_forward_assertion(self):
        transform = transforms.RandomIoUCrop()
        with pytest.raises(
            TypeError,
            match="requires input sample to contain Images or PIL Images, BoundingBoxes and Labels or OneHotLabels",
        ):
            transform(torch.tensor(0))

    def test__transform(self, mocker):
        transform = transforms.RandomIoUCrop()

        image = datapoints.Image(torch.rand(3, 32, 24))
        bboxes = make_bounding_box(format="XYXY", spatial_size=(32, 24), extra_dims=(6,))
        label = datapoints.Label(torch.randint(0, 10, size=(6,)))
        ohe_label = datapoints.OneHotLabel(torch.zeros(6, 10).scatter_(1, label.unsqueeze(1), 1))
        masks = make_detection_mask((32, 24), num_objects=6)

        sample = [image, bboxes, label, ohe_label, masks]

        fn = mocker.patch("torchvision.prototype.transforms.functional.crop", side_effect=lambda x, **params: x)
        is_within_crop_area = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)

        params = dict(top=1, left=2, height=12, width=12, is_within_crop_area=is_within_crop_area)
        transform._get_params = mocker.MagicMock(return_value=params)
        output = transform(sample)

        assert fn.call_count == 3

        expected_calls = [
            mocker.call(image, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
            mocker.call(bboxes, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
            mocker.call(masks, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
        ]

        fn.assert_has_calls(expected_calls)

        expected_within_targets = sum(is_within_crop_area)

        # check number of bboxes vs number of labels:
        output_bboxes = output[1]
        assert isinstance(output_bboxes, datapoints.BoundingBox)
        assert len(output_bboxes) == expected_within_targets

        # check labels
        output_label = output[2]
        assert isinstance(output_label, datapoints.Label)
        assert len(output_label) == expected_within_targets
        torch.testing.assert_close(output_label, label[is_within_crop_area])

        output_ohe_label = output[3]
        assert isinstance(output_ohe_label, datapoints.OneHotLabel)
        torch.testing.assert_close(output_ohe_label, ohe_label[is_within_crop_area])

        output_masks = output[4]
        assert isinstance(output_masks, datapoints.Mask)
        assert len(output_masks) == expected_within_targets


class TestScaleJitter:
    def test__get_params(self, mocker):
        spatial_size = (24, 32)
        target_size = (16, 12)
        scale_range = (0.5, 1.5)

        transform = transforms.ScaleJitter(target_size=target_size, scale_range=scale_range)
        sample = mocker.MagicMock(spec=datapoints.Image, num_channels=3, spatial_size=spatial_size)

        n_samples = 5
        for _ in range(n_samples):

            params = transform._get_params([sample])

            assert "size" in params
            size = params["size"]

            assert isinstance(size, tuple) and len(size) == 2
            height, width = size

            r_min = min(target_size[1] / spatial_size[0], target_size[0] / spatial_size[1]) * scale_range[0]
            r_max = min(target_size[1] / spatial_size[0], target_size[0] / spatial_size[1]) * scale_range[1]

            assert int(spatial_size[0] * r_min) <= height <= int(spatial_size[0] * r_max)
            assert int(spatial_size[1] * r_min) <= width <= int(spatial_size[1] * r_max)

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock()
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.ScaleJitter(
            target_size=(16, 12), interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.ScaleJitter._get_params", return_value=dict(size=size_sentinel)
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.prototype.transforms._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(
            inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


class TestRandomShortestSize:
    @pytest.mark.parametrize("min_size,max_size", [([5, 9], 20), ([5, 9], None)])
    def test__get_params(self, min_size, max_size, mocker):
        spatial_size = (3, 10)

        transform = transforms.RandomShortestSize(min_size=min_size, max_size=max_size)

        sample = mocker.MagicMock(spec=datapoints.Image, num_channels=3, spatial_size=spatial_size)
        params = transform._get_params([sample])

        assert "size" in params
        size = params["size"]

        assert isinstance(size, tuple) and len(size) == 2

        longer = max(size)
        shorter = min(size)
        if max_size is not None:
            assert longer <= max_size
            assert shorter <= max_size
        else:
            assert shorter in min_size

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock()
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.RandomShortestSize(
            min_size=[3, 5, 7], max_size=12, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.RandomShortestSize._get_params",
            return_value=dict(size=size_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.prototype.transforms._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(
            inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


class TestSimpleCopyPaste:
    def create_fake_image(self, mocker, image_type):
        if image_type == PIL.Image.Image:
            return PIL.Image.new("RGB", (32, 32), 123)
        return mocker.MagicMock(spec=image_type)

    def test__extract_image_targets_assertion(self, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, datapoints.Image),
            # labels, bboxes, masks
            mocker.MagicMock(spec=datapoints.Label),
            mocker.MagicMock(spec=datapoints.BoundingBox),
            mocker.MagicMock(spec=datapoints.Mask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=datapoints.BoundingBox),
            mocker.MagicMock(spec=datapoints.Mask),
        ]

        with pytest.raises(TypeError, match="requires input sample to contain equal sized list of Images"):
            transform._extract_image_targets(flat_sample)

    @pytest.mark.parametrize("image_type", [datapoints.Image, PIL.Image.Image, torch.Tensor])
    @pytest.mark.parametrize("label_type", [datapoints.Label, datapoints.OneHotLabel])
    def test__extract_image_targets(self, image_type, label_type, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, image_type),
            self.create_fake_image(mocker, image_type),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=datapoints.BoundingBox),
            mocker.MagicMock(spec=datapoints.Mask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=datapoints.BoundingBox),
            mocker.MagicMock(spec=datapoints.Mask),
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
                ("boxes", datapoints.BoundingBox),
                ("masks", datapoints.Mask),
                ("labels", label_type),
            ]:
                assert key in target
                assert isinstance(target[key], type_)
                assert target[key] in flat_sample

    @pytest.mark.parametrize("label_type", [datapoints.Label, datapoints.OneHotLabel])
    def test__copy_paste(self, label_type):
        image = 2 * torch.ones(3, 32, 32)
        masks = torch.zeros(2, 32, 32)
        masks[0, 3:9, 2:8] = 1
        masks[1, 20:30, 20:30] = 1
        labels = torch.tensor([1, 2])
        blending = True
        resize_interpolation = InterpolationMode.BILINEAR
        antialias = None
        if label_type == datapoints.OneHotLabel:
            labels = torch.nn.functional.one_hot(labels, num_classes=5)
        target = {
            "boxes": datapoints.BoundingBox(
                torch.tensor([[2.0, 3.0, 8.0, 9.0], [20.0, 20.0, 30.0, 30.0]]), format="XYXY", spatial_size=(32, 32)
            ),
            "masks": datapoints.Mask(masks),
            "labels": label_type(labels),
        }

        paste_image = 10 * torch.ones(3, 32, 32)
        paste_masks = torch.zeros(2, 32, 32)
        paste_masks[0, 13:19, 12:18] = 1
        paste_masks[1, 15:19, 1:8] = 1
        paste_labels = torch.tensor([3, 4])
        if label_type == datapoints.OneHotLabel:
            paste_labels = torch.nn.functional.one_hot(paste_labels, num_classes=5)
        paste_target = {
            "boxes": datapoints.BoundingBox(
                torch.tensor([[12.0, 13.0, 19.0, 18.0], [1.0, 15.0, 8.0, 19.0]]), format="XYXY", spatial_size=(32, 32)
            ),
            "masks": datapoints.Mask(paste_masks),
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
        if label_type == datapoints.OneHotLabel:
            expected_labels = torch.nn.functional.one_hot(expected_labels, num_classes=5)
        torch.testing.assert_close(output_target["labels"], label_type(expected_labels))

        assert output_target["masks"].shape == (4, 32, 32)
        torch.testing.assert_close(output_target["masks"][:2, :], target["masks"])
        torch.testing.assert_close(output_target["masks"][2:, :], paste_target["masks"])


class TestFixedSizeCrop:
    def test__get_params(self, mocker):
        crop_size = (7, 7)
        batch_shape = (10,)
        spatial_size = (11, 5)

        transform = transforms.FixedSizeCrop(size=crop_size)

        flat_inputs = [
            make_image(size=spatial_size, color_space="RGB"),
            make_bounding_box(
                format=datapoints.BoundingBoxFormat.XYXY, spatial_size=spatial_size, extra_dims=batch_shape
            ),
        ]
        params = transform._get_params(flat_inputs)

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

    @pytest.mark.parametrize("needs", list(itertools.product((False, True), repeat=2)))
    def test__transform(self, mocker, needs):
        fill_sentinel = 12
        padding_mode_sentinel = mocker.MagicMock()

        transform = transforms.FixedSizeCrop((-1, -1), fill=fill_sentinel, padding_mode=padding_mode_sentinel)
        transform._transformed_types = (mocker.MagicMock,)
        mocker.patch("torchvision.prototype.transforms._geometry.has_all", return_value=True)
        mocker.patch("torchvision.prototype.transforms._geometry.has_any", return_value=True)

        needs_crop, needs_pad = needs
        top_sentinel = mocker.MagicMock()
        left_sentinel = mocker.MagicMock()
        height_sentinel = mocker.MagicMock()
        width_sentinel = mocker.MagicMock()
        is_valid = mocker.MagicMock() if needs_crop else None
        padding_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop._get_params",
            return_value=dict(
                needs_crop=needs_crop,
                top=top_sentinel,
                left=left_sentinel,
                height=height_sentinel,
                width=width_sentinel,
                is_valid=is_valid,
                padding=padding_sentinel,
                needs_pad=needs_pad,
            ),
        )

        inpt_sentinel = mocker.MagicMock()

        mock_crop = mocker.patch("torchvision.prototype.transforms._geometry.F.crop")
        mock_pad = mocker.patch("torchvision.prototype.transforms._geometry.F.pad")
        transform(inpt_sentinel)

        if needs_crop:
            mock_crop.assert_called_once_with(
                inpt_sentinel,
                top=top_sentinel,
                left=left_sentinel,
                height=height_sentinel,
                width=width_sentinel,
            )
        else:
            mock_crop.assert_not_called()

        if needs_pad:
            # If we cropped before, the input to F.pad is no longer inpt_sentinel. Thus, we can't use
            # `MagicMock.assert_called_once_with` and have to perform the checks manually
            mock_pad.assert_called_once()
            args, kwargs = mock_pad.call_args
            if not needs_crop:
                assert args[0] is inpt_sentinel
            assert args[1] is padding_sentinel
            fill_sentinel = transforms._utils._convert_fill_arg(fill_sentinel)
            assert kwargs == dict(fill=fill_sentinel, padding_mode=padding_mode_sentinel)
        else:
            mock_pad.assert_not_called()

    def test__transform_culling(self, mocker):
        batch_size = 10
        spatial_size = (10, 10)

        is_valid = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop._get_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=spatial_size[0],
                width=spatial_size[1],
                is_valid=is_valid,
                needs_pad=False,
            ),
        )

        bounding_boxes = make_bounding_box(
            format=datapoints.BoundingBoxFormat.XYXY, spatial_size=spatial_size, extra_dims=(batch_size,)
        )
        masks = make_detection_mask(size=spatial_size, extra_dims=(batch_size,))
        labels = make_label(extra_dims=(batch_size,))

        transform = transforms.FixedSizeCrop((-1, -1))
        mocker.patch("torchvision.prototype.transforms._geometry.has_all", return_value=True)
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

    def test__transform_bounding_box_clamping(self, mocker):
        batch_size = 3
        spatial_size = (10, 10)

        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop._get_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=spatial_size[0],
                width=spatial_size[1],
                is_valid=torch.full((batch_size,), fill_value=True),
                needs_pad=False,
            ),
        )

        bounding_box = make_bounding_box(
            format=datapoints.BoundingBoxFormat.XYXY, spatial_size=spatial_size, extra_dims=(batch_size,)
        )
        mock = mocker.patch("torchvision.prototype.transforms._geometry.F.clamp_bounding_box")

        transform = transforms.FixedSizeCrop((-1, -1))
        mocker.patch("torchvision.prototype.transforms._geometry.has_all", return_value=True)
        mocker.patch("torchvision.prototype.transforms._geometry.has_any", return_value=True)

        transform(bounding_box)

        mock.assert_called_once()


class TestLinearTransformation:
    def test_assertions(self):
        with pytest.raises(ValueError, match="transformation_matrix should be square"):
            transforms.LinearTransformation(torch.rand(2, 3), torch.rand(5))

        with pytest.raises(ValueError, match="mean_vector should have the same length"):
            transforms.LinearTransformation(torch.rand(3, 3), torch.rand(5))

    @pytest.mark.parametrize(
        "inpt",
        [
            122 * torch.ones(1, 3, 8, 8),
            122.0 * torch.ones(1, 3, 8, 8),
            datapoints.Image(122 * torch.ones(1, 3, 8, 8)),
            PIL.Image.new("RGB", (8, 8), (122, 122, 122)),
        ],
    )
    def test__transform(self, inpt):

        v = 121 * torch.ones(3 * 8 * 8)
        m = torch.ones(3 * 8 * 8, 3 * 8 * 8)
        transform = transforms.LinearTransformation(m, v)

        if isinstance(inpt, PIL.Image.Image):
            with pytest.raises(TypeError, match="LinearTransformation does not work on PIL Images"):
                transform(inpt)
        else:
            output = transform(inpt)
            assert isinstance(output, torch.Tensor)
            assert output.unique() == 3 * 8 * 8
            assert output.dtype == inpt.dtype


class TestLabelToOneHot:
    def test__transform(self):
        categories = ["apple", "pear", "pineapple"]
        labels = datapoints.Label(torch.tensor([0, 1, 2, 1]), categories=categories)
        transform = transforms.LabelToOneHot()
        ohe_labels = transform(labels)
        assert isinstance(ohe_labels, datapoints.OneHotLabel)
        assert ohe_labels.shape == (4, 3)
        assert ohe_labels.categories == labels.categories == categories


class TestRandomResize:
    def test__get_params(self):
        min_size = 3
        max_size = 6

        transform = transforms.RandomResize(min_size=min_size, max_size=max_size)

        for _ in range(10):
            params = transform._get_params([])

            assert isinstance(params["size"], list) and len(params["size"]) == 1
            size = params["size"][0]

            assert min_size <= size < max_size

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock()
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.RandomResize(
            min_size=-1, max_size=-1, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.RandomResize._get_params",
            return_value=dict(size=size_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock_resize = mocker.patch("torchvision.prototype.transforms._geometry.F.resize")
        transform(inpt_sentinel)

        mock_resize.assert_called_with(
            inpt_sentinel, size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


class TestToDtype:
    @pytest.mark.parametrize(
        ("dtype", "expected_dtypes"),
        [
            (
                torch.float64,
                {
                    datapoints.Video: torch.float64,
                    datapoints.Image: torch.float64,
                    datapoints.BoundingBox: torch.float64,
                },
            ),
            (
                {datapoints.Video: torch.int32, datapoints.Image: torch.float32, datapoints.BoundingBox: torch.float64},
                {datapoints.Video: torch.int32, datapoints.Image: torch.float32, datapoints.BoundingBox: torch.float64},
            ),
        ],
    )
    def test_call(self, dtype, expected_dtypes):
        sample = dict(
            video=make_video(dtype=torch.int64),
            image=make_image(dtype=torch.uint8),
            bounding_box=make_bounding_box(format=datapoints.BoundingBoxFormat.XYXY, dtype=torch.float32),
            str="str",
            int=0,
        )

        transform = transforms.ToDtype(dtype)
        transformed_sample = transform(sample)

        for key, value in sample.items():
            value_type = type(value)
            transformed_value = transformed_sample[key]

            # make sure the transformation retains the type
            assert isinstance(transformed_value, value_type)

            if isinstance(value, torch.Tensor):
                assert transformed_value.dtype is expected_dtypes[value_type]
            else:
                assert transformed_value is value

    @pytest.mark.filterwarnings("error")
    def test_plain_tensor_call(self):
        tensor = torch.empty((), dtype=torch.float32)
        transform = transforms.ToDtype({torch.Tensor: torch.float64})

        assert transform(tensor).dtype is torch.float64

    @pytest.mark.parametrize("other_type", [datapoints.Image, datapoints.Video])
    def test_plain_tensor_warning(self, other_type):
        with pytest.warns(UserWarning, match=re.escape("`torch.Tensor` will *not* be transformed")):
            transforms.ToDtype(dtype={torch.Tensor: torch.float32, other_type: torch.float64})


class TestPermuteDimensions:
    @pytest.mark.parametrize(
        ("dims", "inverse_dims"),
        [
            (
                {datapoints.Image: (2, 1, 0), datapoints.Video: None},
                {datapoints.Image: (2, 1, 0), datapoints.Video: None},
            ),
            (
                {datapoints.Image: (2, 1, 0), datapoints.Video: (1, 2, 3, 0)},
                {datapoints.Image: (2, 1, 0), datapoints.Video: (3, 0, 1, 2)},
            ),
        ],
    )
    def test_call(self, dims, inverse_dims):
        sample = dict(
            image=make_image(),
            bounding_box=make_bounding_box(format=datapoints.BoundingBoxFormat.XYXY),
            video=make_video(),
            str="str",
            int=0,
        )

        transform = transforms.PermuteDimensions(dims)
        transformed_sample = transform(sample)

        for key, value in sample.items():
            value_type = type(value)
            transformed_value = transformed_sample[key]

            if check_type(
                value, (datapoints.Image, torchvision.prototype.transforms.utils.is_simple_tensor, datapoints.Video)
            ):
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

    @pytest.mark.parametrize("other_type", [datapoints.Image, datapoints.Video])
    def test_plain_tensor_warning(self, other_type):
        with pytest.warns(UserWarning, match=re.escape("`torch.Tensor` will *not* be transformed")):
            transforms.PermuteDimensions(dims={torch.Tensor: (0, 1), other_type: (1, 0)})


class TestTransposeDimensions:
    @pytest.mark.parametrize(
        "dims",
        [
            (-1, -2),
            {datapoints.Image: (1, 2), datapoints.Video: None},
        ],
    )
    def test_call(self, dims):
        sample = dict(
            image=make_image(),
            bounding_box=make_bounding_box(format=datapoints.BoundingBoxFormat.XYXY),
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
            if check_type(
                value, (datapoints.Image, torchvision.prototype.transforms.utils.is_simple_tensor, datapoints.Video)
            ):
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

    @pytest.mark.parametrize("other_type", [datapoints.Image, datapoints.Video])
    def test_plain_tensor_warning(self, other_type):
        with pytest.warns(UserWarning, match=re.escape("`torch.Tensor` will *not* be transformed")):
            transforms.TransposeDimensions(dims={torch.Tensor: (0, 1), other_type: (1, 0)})


class TestUniformTemporalSubsample:
    @pytest.mark.parametrize(
        "inpt",
        [
            torch.zeros(10, 3, 8, 8),
            torch.zeros(1, 10, 3, 8, 8),
            datapoints.Video(torch.zeros(1, 10, 3, 8, 8)),
        ],
    )
    def test__transform(self, inpt):
        num_samples = 5
        transform = transforms.UniformTemporalSubsample(num_samples)

        output = transform(inpt)
        assert type(output) is type(inpt)
        assert output.shape[-4] == num_samples
        assert output.dtype == inpt.dtype
