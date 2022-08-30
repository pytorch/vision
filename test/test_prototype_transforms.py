import itertools

import numpy as np

import PIL.Image

import pytest
import torch
from common_utils import assert_equal, cpu_and_gpu
from test_prototype_transforms_functional import (
    make_bounding_box,
    make_bounding_boxes,
    make_image,
    make_images,
    make_label,
    make_one_hot_labels,
    make_segmentation_mask,
)
from torchvision.ops.boxes import box_iou
from torchvision.prototype import features, transforms
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor, to_pil_image


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
        transforms.ConvertImageDtype(),
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
                    dict(
                        image=features.Image.new_like(image, image.unsqueeze(0), dtype=torch.float),
                        one_hot_label=features.OneHotLabel.new_like(
                            one_hot_label, one_hot_label.unsqueeze(0), dtype=torch.float
                        ),
                    )
                    for image, one_hot_label in itertools.product(make_images(), make_one_hot_labels())
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

        # add other data that should bypass and wont raise any error
        input_copy = dict(input)
        input_copy["path"] = "/path/to/somewhere"
        input_copy["num"] = 1234
        transform(input_copy)

        # Check if we raise an error if sample contains bbox or mask or label
        err_msg = "does not support bounding boxes, segmentation masks and plain labels"
        input_copy = dict(input)
        for unsup_data in [make_label(), make_bounding_box(format="XYXY"), make_segmentation_mask()]:
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
                            features.ColorSpace.GRAY,
                            features.ColorSpace.RGB,
                        ],
                        dtypes=[torch.uint8],
                        extra_dims=[(4,)],
                    )
                    for fn in [
                        make_images,
                        make_vanilla_tensor_images,
                        make_pil_images,
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
                    fn(color_spaces=[features.ColorSpace.RGB], dtypes=[torch.float32])
                    for fn in [
                        make_images,
                        make_vanilla_tensor_images,
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
                ),
            )
        ]
    )
    def test_random_resized_crop(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transforms.ConvertColorSpace(color_space=new_color_space, old_color_space=old_color_space),
                itertools.chain.from_iterable(
                    [
                        fn(color_spaces=[old_color_space])
                        for fn in (
                            make_images,
                            make_vanilla_tensor_images,
                            make_pil_images,
                        )
                    ]
                ),
            )
            for old_color_space, new_color_space in itertools.product(
                [
                    features.ColorSpace.GRAY,
                    features.ColorSpace.GRAY_ALPHA,
                    features.ColorSpace.RGB,
                    features.ColorSpace.RGB_ALPHA,
                ],
                repeat=2,
            )
        ]
    )
    def test_convert_color_space(self, transform, input):
        transform(input)

    def test_convert_color_space_unsupported_types(self):
        transform = transforms.ConvertColorSpace(
            color_space=features.ColorSpace.RGB, old_color_space=features.ColorSpace.GRAY
        )

        for inpt in [make_bounding_box(format="XYXY"), make_segmentation_mask()]:
            output = transform(inpt)
            assert output is inpt


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

    def test_features_image(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(features.Image(input))

        assert_equal(features.Image(expected), actual)

    def test_features_segmentation_mask(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(features.SegmentationMask(input))

        assert_equal(features.SegmentationMask(expected), actual)

    def test_features_bounding_box(self, p):
        input = features.BoundingBox([0, 0, 5, 5], format=features.BoundingBoxFormat.XYXY, image_size=(10, 10))
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(input)

        expected_image_tensor = torch.tensor([5, 0, 10, 5]) if p == 1.0 else input
        expected = features.BoundingBox.new_like(input, data=expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.image_size == expected.image_size


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

    def test_features_image(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(features.Image(input))

        assert_equal(features.Image(expected), actual)

    def test_features_segmentation_mask(self, p):
        input, expected = self.input_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(features.SegmentationMask(input))

        assert_equal(features.SegmentationMask(expected), actual)

    def test_features_bounding_box(self, p):
        input = features.BoundingBox([0, 0, 5, 5], format=features.BoundingBoxFormat.XYXY, image_size=(10, 10))
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(input)

        expected_image_tensor = torch.tensor([0, 5, 5, 10]) if p == 1.0 else input
        expected = features.BoundingBox.new_like(input, data=expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.image_size == expected.image_size


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
        inpt = mocker.MagicMock(spec=features.Image)
        _ = transform(inpt)

        fn.assert_called_once_with(inpt, padding=padding, fill=fill, padding_mode=padding_mode)


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

        image = mocker.MagicMock(spec=features.Image)
        h, w = image.image_size = (24, 32)

        params = transform._get_params(image)

        assert params["fill"] == fill
        assert len(params["padding"]) == 4
        assert 0 <= params["padding"][0] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][1] <= (side_range[1] - 1) * h
        assert 0 <= params["padding"][2] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][3] <= (side_range[1] - 1) * h

    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    def test__transform(self, fill, side_range, mocker):
        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (24, 32)

        transform = transforms.RandomZoomOut(fill=fill, side_range=side_range, p=1)

        fn = mocker.patch("torchvision.prototype.transforms.functional.pad")
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        torch.rand(1)  # random apply changes random state
        params = transform._get_params(inpt)

        fn.assert_called_once_with(inpt, **params)


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
        inpt = mocker.MagicMock(spec=features.Image)
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params(inpt)

        fn.assert_called_once_with(inpt, **params, interpolation=interpolation, expand=expand, fill=fill, center=center)

    @pytest.mark.parametrize("angle", [34, -87])
    @pytest.mark.parametrize("expand", [False, True])
    def test_boundingbox_image_size(self, angle, expand):
        # Specific test for BoundingBox.rotate
        bbox = features.BoundingBox(
            torch.tensor([1, 2, 3, 4]), format=features.BoundingBoxFormat.XYXY, image_size=(32, 32)
        )
        img = features.Image(torch.rand(1, 3, 32, 32))

        out_img = img.rotate(angle, expand=expand)
        out_bbox = bbox.rotate(angle, expand=expand)

        assert out_img.image_size == out_bbox.image_size


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
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)
        h, w = image.image_size

        transform = transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear)
        params = transform._get_params(image)

        if not isinstance(degrees, (list, tuple)):
            assert -degrees <= params["angle"] <= degrees
        else:
            assert degrees[0] <= params["angle"] <= degrees[1]

        if translate is not None:
            w_max = int(round(translate[0] * w))
            h_max = int(round(translate[1] * h))
            assert -w_max <= params["translations"][0] <= w_max
            assert -h_max <= params["translations"][1] <= h_max
        else:
            assert params["translations"] == (0, 0)

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
        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (24, 32)

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params(inpt)

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
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)
        h, w = image.image_size

        transform = transforms.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed)
        params = transform._get_params(image)

        if padding is not None:
            if isinstance(padding, int):
                h += 2 * padding
                w += 2 * padding
            elif isinstance(padding, list) and len(padding) == 2:
                w += 2 * padding[0]
                h += 2 * padding[1]
            elif isinstance(padding, list) and len(padding) == 4:
                w += padding[0] + padding[2]
                h += padding[1] + padding[3]

        expected_input_width = w
        expected_input_height = h

        if pad_if_needed:
            if w < size[1]:
                w += 2 * (size[1] - w)
            if h < size[0]:
                h += 2 * (size[0] - h)

        assert 0 <= params["top"] <= h - size[0] + 1
        assert 0 <= params["left"] <= w - size[1] + 1
        assert params["height"] == size[0]
        assert params["width"] == size[1]
        assert params["input_width"] == expected_input_width
        assert params["input_height"] == expected_input_height

    @pytest.mark.parametrize("padding", [None, 1, [2, 3], [1, 2, 3, 4]])
    @pytest.mark.parametrize("pad_if_needed", [False, True])
    @pytest.mark.parametrize("fill", [False, True])
    @pytest.mark.parametrize("padding_mode", ["constant", "edge"])
    def test__transform(self, padding, pad_if_needed, fill, padding_mode, mocker):
        output_size = [10, 12]
        transform = transforms.RandomCrop(
            output_size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode
        )

        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (32, 32)

        expected = mocker.MagicMock(spec=features.Image)
        expected.num_channels = 3
        if isinstance(padding, int):
            expected.image_size = (inpt.image_size[0] + padding, inpt.image_size[1] + padding)
        elif isinstance(padding, list):
            expected.image_size = (
                inpt.image_size[0] + sum(padding[0::2]),
                inpt.image_size[1] + sum(padding[1::2]),
            )
        else:
            expected.image_size = inpt.image_size
        _ = mocker.patch("torchvision.prototype.transforms.functional.pad", return_value=expected)
        fn_crop = mocker.patch("torchvision.prototype.transforms.functional.crop")

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params(inpt)
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

        with pytest.raises(TypeError, match="sigma should be a single float or a list/tuple with length 2"):
            transforms.GaussianBlur(3, sigma=[1, 2, 3])

        with pytest.raises(ValueError, match="If sigma is a single number, it must be positive"):
            transforms.GaussianBlur(3, sigma=-1.0)

        with pytest.raises(ValueError, match="sigma values should be positive and of the form"):
            transforms.GaussianBlur(3, sigma=[2.0, 1.0])

    @pytest.mark.parametrize("sigma", [10.0, [10.0, 12.0]])
    def test__get_params(self, sigma):
        transform = transforms.GaussianBlur(3, sigma=sigma)
        params = transform._get_params(None)

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
            assert transform.kernel_size == (kernel_size, kernel_size)

        if isinstance(sigma, (tuple, list)):
            assert transform.sigma == sigma
        else:
            assert transform.sigma == (sigma, sigma)

        fn = mocker.patch("torchvision.prototype.transforms.functional.gaussian_blur")
        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (24, 32)

        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        params = transform._get_params(inpt)

        fn.assert_called_once_with(inpt, **params)


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
        inpt = mocker.MagicMock(spec=features.Image)
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
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)

        params = transform._get_params(image)

        h, w = image.image_size
        assert len(params["startpoints"]) == 4
        for x, y in params["startpoints"]:
            assert x in (0, w - 1)
            assert y in (0, h - 1)

        assert len(params["endpoints"]) == 4
        for (x, y), name in zip(params["endpoints"], ["tl", "tr", "br", "bl"]):
            if "t" in name:
                assert 0 <= y <= int(dscale * h // 2), (x, y, name)
            if "b" in name:
                assert h - int(dscale * h // 2) - 1 <= y <= h, (x, y, name)
            if "l" in name:
                assert 0 <= x <= int(dscale * w // 2), (x, y, name)
            if "r" in name:
                assert w - int(dscale * w // 2) - 1 <= x <= w, (x, y, name)

    @pytest.mark.parametrize("distortion_scale", [0.1, 0.7])
    def test__transform(self, distortion_scale, mocker):
        interpolation = InterpolationMode.BILINEAR
        fill = 12
        transform = transforms.RandomPerspective(distortion_scale, fill=fill, interpolation=interpolation)

        fn = mocker.patch("torchvision.prototype.transforms.functional.perspective")
        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (24, 32)
        # vfdev-5, Feature Request: let's store params as Transform attribute
        # This could be also helpful for users
        # Otherwise, we can mock transform._get_params
        torch.manual_seed(12)
        _ = transform(inpt)
        torch.manual_seed(12)
        torch.rand(1)  # random apply changes random state
        params = transform._get_params(inpt)

        fn.assert_called_once_with(inpt, **params, fill=fill, interpolation=interpolation)


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
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)

        params = transform._get_params(image)

        h, w = image.image_size
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
        inpt = mocker.MagicMock(spec=features.Image)
        inpt.num_channels = 3
        inpt.image_size = (24, 32)

        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock()
        _ = transform(inpt)
        params = transform._get_params(inpt)
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

        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)

        transform = transforms.RandomErasing(value=[1, 2, 3, 4])

        with pytest.raises(ValueError, match="If value is a sequence, it should have either a single value"):
            transform._get_params(image)

    @pytest.mark.parametrize("value", [5.0, [1, 2, 3], "random"])
    def test__get_params(self, value, mocker):
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)

        transform = transforms.RandomErasing(value=value)
        params = transform._get_params(image)

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

        assert 0 <= i <= image.image_size[0] - h
        assert 0 <= j <= image.image_size[1] - w

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
                inpt_sentinel, i=i_sentinel, j=j_sentinel, h=h_sentinel, w=w_sentinel, v=v_sentinel
            )
        else:
            mock.assert_not_called()
            assert output is inpt_sentinel


class TestTransform:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
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
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch(
            "torchvision.prototype.transforms.functional.to_image_tensor",
            return_value=torch.rand(1, 3, 8, 8),
        )

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImageTensor()
        transform(inpt)
        if inpt_type in (features.BoundingBox, features.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt)


class TestToImagePIL:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.prototype.transforms.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImagePIL()
        transform(inpt)
        if inpt_type in (features.BoundingBox, PIL.Image.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToPILImage:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.prototype.transforms.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToPILImage()
        transform(inpt)
        if inpt_type in (PIL.Image.Image, features.BoundingBox, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToTensor:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.functional.to_tensor")

        inpt = mocker.MagicMock(spec=inpt_type)
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            transform = transforms.ToTensor()
        transform(inpt)
        if inpt_type in (features.Image, torch.Tensor, features.BoundingBox, str, int):
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
        "trfms", [[transforms.Pad(2), transforms.RandomCrop(28)], [lambda x: 2.0 * x, transforms.RandomCrop(28)]]
    )
    def test_ctor(self, transform_cls, trfms):
        c = transform_cls(trfms)
        inpt = torch.rand(1, 3, 32, 32)
        output = c(inpt)
        assert isinstance(output, torch.Tensor)


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
        image = mocker.MagicMock(spec=features.Image)
        image.num_channels = 3
        image.image_size = (24, 32)
        bboxes = features.BoundingBox(
            torch.tensor([[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]),
            format="XYXY",
            image_size=image.image_size,
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

            orig_h = image.image_size[0]
            orig_w = image.image_size[1]
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
        image = features.Image(torch.rand(1, 3, 4, 4))
        bboxes = features.BoundingBox(torch.tensor([[1, 1, 2, 2]]), format="XYXY", image_size=(4, 4))
        label = features.Label(torch.tensor([1]))
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

        image = features.Image(torch.rand(3, 32, 24))
        bboxes = make_bounding_box(format="XYXY", image_size=(32, 24), extra_dims=(6,))
        label = features.Label(torch.randint(0, 10, size=(6,)))
        ohe_label = features.OneHotLabel(torch.zeros(6, 10).scatter_(1, label.unsqueeze(1), 1))
        masks = make_segmentation_mask((32, 24))
        ohe_masks = features.SegmentationMask(torch.randint(0, 2, size=(6, 32, 24)))
        sample = [image, bboxes, label, ohe_label, masks, ohe_masks]

        fn = mocker.patch("torchvision.prototype.transforms.functional.crop", side_effect=lambda x, **params: x)
        is_within_crop_area = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)

        params = dict(top=1, left=2, height=12, width=12, is_within_crop_area=is_within_crop_area)
        transform._get_params = mocker.MagicMock(return_value=params)
        output = transform(sample)

        assert fn.call_count == 4

        expected_calls = [
            mocker.call(image, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
            mocker.call(bboxes, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
            mocker.call(masks, top=params["top"], left=params["left"], height=params["height"], width=params["width"]),
            mocker.call(
                ohe_masks, top=params["top"], left=params["left"], height=params["height"], width=params["width"]
            ),
        ]

        fn.assert_has_calls(expected_calls)

        expected_within_targets = sum(is_within_crop_area)

        # check number of bboxes vs number of labels:
        output_bboxes = output[1]
        assert isinstance(output_bboxes, features.BoundingBox)
        assert len(output_bboxes) == expected_within_targets

        # check labels
        output_label = output[2]
        assert isinstance(output_label, features.Label)
        assert len(output_label) == expected_within_targets
        torch.testing.assert_close(output_label, label[is_within_crop_area])

        output_ohe_label = output[3]
        assert isinstance(output_ohe_label, features.OneHotLabel)
        torch.testing.assert_close(output_ohe_label, ohe_label[is_within_crop_area])

        output_masks = output[4]
        assert isinstance(output_masks, features.SegmentationMask)
        assert output_masks.shape[:-2] == masks.shape[:-2]

        output_ohe_masks = output[5]
        assert isinstance(output_ohe_masks, features.SegmentationMask)
        assert len(output_ohe_masks) == expected_within_targets


class TestScaleJitter:
    def test__get_params(self, mocker):
        image_size = (24, 32)
        target_size = (16, 12)
        scale_range = (0.5, 1.5)

        transform = transforms.ScaleJitter(target_size=target_size, scale_range=scale_range)

        sample = mocker.MagicMock(spec=features.Image, num_channels=3, image_size=image_size)
        params = transform._get_params(sample)

        assert "size" in params
        size = params["size"]

        assert isinstance(size, tuple) and len(size) == 2
        height, width = size

        assert int(target_size[0] * scale_range[0]) <= height <= int(target_size[0] * scale_range[1])
        assert int(target_size[1] * scale_range[0]) <= width <= int(target_size[1] * scale_range[1])

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock()

        transform = transforms.ScaleJitter(target_size=(16, 12), interpolation=interpolation_sentinel)
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.ScaleJitter._get_params", return_value=dict(size=size_sentinel)
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.prototype.transforms._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel)


class TestRandomShortestSize:
    def test__get_params(self, mocker):
        image_size = (3, 10)
        min_size = [5, 9]
        max_size = 20

        transform = transforms.RandomShortestSize(min_size=min_size, max_size=max_size)

        sample = mocker.MagicMock(spec=features.Image, num_channels=3, image_size=image_size)
        params = transform._get_params(sample)

        assert "size" in params
        size = params["size"]

        assert isinstance(size, tuple) and len(size) == 2

        longer = max(size)
        assert longer <= max_size

        shorter = min(size)
        if longer == max_size:
            assert shorter <= max_size
        else:
            assert shorter in min_size

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock()

        transform = transforms.RandomShortestSize(min_size=[3, 5, 7], max_size=12, interpolation=interpolation_sentinel)
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.prototype.transforms._geometry.RandomShortestSize._get_params",
            return_value=dict(size=size_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.prototype.transforms._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel)


class TestSimpleCopyPaste:
    def create_fake_image(self, mocker, image_type):
        if image_type == PIL.Image.Image:
            return PIL.Image.new("RGB", (32, 32), 123)
        return mocker.MagicMock(spec=image_type)

    def test__extract_image_targets_assertion(self, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, features.Image),
            # labels, bboxes, masks
            mocker.MagicMock(spec=features.Label),
            mocker.MagicMock(spec=features.BoundingBox),
            mocker.MagicMock(spec=features.SegmentationMask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=features.BoundingBox),
            mocker.MagicMock(spec=features.SegmentationMask),
        ]

        with pytest.raises(TypeError, match="requires input sample to contain equal sized list of Images"):
            transform._extract_image_targets(flat_sample)

    @pytest.mark.parametrize("image_type", [features.Image, PIL.Image.Image, torch.Tensor])
    @pytest.mark.parametrize("label_type", [features.Label, features.OneHotLabel])
    def test__extract_image_targets(self, image_type, label_type, mocker):
        transform = transforms.SimpleCopyPaste()

        flat_sample = [
            # images, batch size = 2
            self.create_fake_image(mocker, image_type),
            self.create_fake_image(mocker, image_type),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=features.BoundingBox),
            mocker.MagicMock(spec=features.SegmentationMask),
            # labels, bboxes, masks
            mocker.MagicMock(spec=label_type),
            mocker.MagicMock(spec=features.BoundingBox),
            mocker.MagicMock(spec=features.SegmentationMask),
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
                ("boxes", features.BoundingBox),
                ("masks", features.SegmentationMask),
                ("labels", label_type),
            ]:
                assert key in target
                assert isinstance(target[key], type_)
                assert target[key] in flat_sample

    @pytest.mark.parametrize("label_type", [features.Label, features.OneHotLabel])
    def test__copy_paste(self, label_type):
        image = 2 * torch.ones(3, 32, 32)
        masks = torch.zeros(2, 32, 32)
        masks[0, 3:9, 2:8] = 1
        masks[1, 20:30, 20:30] = 1
        labels = torch.tensor([1, 2])
        if label_type == features.OneHotLabel:
            labels = torch.nn.functional.one_hot(labels, num_classes=5)
        target = {
            "boxes": features.BoundingBox(
                torch.tensor([[2.0, 3.0, 8.0, 9.0], [20.0, 20.0, 30.0, 30.0]]), format="XYXY", image_size=(32, 32)
            ),
            "masks": features.SegmentationMask(masks),
            "labels": label_type(labels),
        }

        paste_image = 10 * torch.ones(3, 32, 32)
        paste_masks = torch.zeros(2, 32, 32)
        paste_masks[0, 13:19, 12:18] = 1
        paste_masks[1, 15:19, 1:8] = 1
        paste_labels = torch.tensor([3, 4])
        if label_type == features.OneHotLabel:
            paste_labels = torch.nn.functional.one_hot(paste_labels, num_classes=5)
        paste_target = {
            "boxes": features.BoundingBox(
                torch.tensor([[12.0, 13.0, 19.0, 18.0], [1.0, 15.0, 8.0, 19.0]]), format="XYXY", image_size=(32, 32)
            ),
            "masks": features.SegmentationMask(paste_masks),
            "labels": label_type(paste_labels),
        }

        transform = transforms.SimpleCopyPaste()
        random_selection = torch.tensor([0, 1])
        output_image, output_target = transform._copy_paste(image, target, paste_image, paste_target, random_selection)

        assert output_image.unique().tolist() == [2, 10]
        assert output_target["boxes"].shape == (4, 4)
        torch.testing.assert_close(output_target["boxes"][:2, :], target["boxes"])
        torch.testing.assert_close(output_target["boxes"][2:, :], paste_target["boxes"])

        expected_labels = torch.tensor([1, 2, 3, 4])
        if label_type == features.OneHotLabel:
            expected_labels = torch.nn.functional.one_hot(expected_labels, num_classes=5)
        torch.testing.assert_close(output_target["labels"], label_type(expected_labels))

        assert output_target["masks"].shape == (4, 32, 32)
        torch.testing.assert_close(output_target["masks"][:2, :], target["masks"])
        torch.testing.assert_close(output_target["masks"][2:, :], paste_target["masks"])


class TestFixedSizeCrop:
    def test__get_params(self, mocker):
        crop_size = (7, 7)
        batch_shape = (10,)
        image_size = (11, 5)

        transform = transforms.FixedSizeCrop(size=crop_size)

        sample = dict(
            image=make_image(size=image_size, color_space=features.ColorSpace.RGB),
            bounding_boxes=make_bounding_box(
                format=features.BoundingBoxFormat.XYXY, image_size=image_size, extra_dims=batch_shape
            ),
        )
        params = transform._get_params(sample)

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
        fill_sentinel = mocker.MagicMock()
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
            assert kwargs == dict(fill=fill_sentinel, padding_mode=padding_mode_sentinel)
        else:
            mock_pad.assert_not_called()

    def test__transform_culling(self, mocker):
        batch_size = 10
        image_size = (10, 10)

        is_valid = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop._get_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=image_size[0],
                width=image_size[1],
                is_valid=is_valid,
                needs_pad=False,
            ),
        )

        bounding_boxes = make_bounding_box(
            format=features.BoundingBoxFormat.XYXY, image_size=image_size, extra_dims=(batch_size,)
        )
        segmentation_masks = make_segmentation_mask(size=image_size, extra_dims=(batch_size,))
        labels = make_label(size=(batch_size,))

        transform = transforms.FixedSizeCrop((-1, -1))
        mocker.patch("torchvision.prototype.transforms._geometry.has_all", return_value=True)
        mocker.patch("torchvision.prototype.transforms._geometry.has_any", return_value=True)

        output = transform(
            dict(
                bounding_boxes=bounding_boxes,
                segmentation_masks=segmentation_masks,
                labels=labels,
            )
        )

        assert_equal(output["bounding_boxes"], bounding_boxes[is_valid])
        assert_equal(output["segmentation_masks"], segmentation_masks[is_valid])
        assert_equal(output["labels"], labels[is_valid])

    def test__transform_bounding_box_clamping(self, mocker):
        batch_size = 3
        image_size = (10, 10)

        mocker.patch(
            "torchvision.prototype.transforms._geometry.FixedSizeCrop._get_params",
            return_value=dict(
                needs_crop=True,
                top=0,
                left=0,
                height=image_size[0],
                width=image_size[1],
                is_valid=torch.full((batch_size,), fill_value=True),
                needs_pad=False,
            ),
        )

        bounding_box = make_bounding_box(
            format=features.BoundingBoxFormat.XYXY, image_size=image_size, extra_dims=(batch_size,)
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
            features.Image(122 * torch.ones(1, 3, 8, 8)),
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
        labels = features.Label(torch.tensor([0, 1, 2, 1]), categories=categories)
        transform = transforms.LabelToOneHot()
        ohe_labels = transform(labels)
        assert isinstance(ohe_labels, features.OneHotLabel)
        assert ohe_labels.shape == (4, 3)
        assert ohe_labels.categories == labels.categories == categories
