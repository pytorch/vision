import itertools

import numpy as np

import PIL.Image

import pytest
import torch
from common_utils import assert_equal
from test_prototype_transforms_functional import (
    make_bounding_box,
    make_bounding_boxes,
    make_images,
    make_label,
    make_one_hot_labels,
    make_segmentation_mask,
)
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
                transforms.ConvertImageColorSpace(color_space=new_color_space, old_color_space=old_color_space),
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
    def test_convert_image_color_space(self, transform, input):
        transform(input)


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
            fn.call_count == 0


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

    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize(
        "inpt_type",
        [
            (torch.Tensor, {"shape": (3, 24, 32)}),
            (PIL.Image.Image, {"size": (24, 32), "mode": "RGB"}),
        ],
    )
    def test__transform(self, p, inpt_type, mocker):
        value = 1.0
        transform = transforms.RandomErasing(p=p, value=value)

        inpt = mocker.MagicMock(spec=inpt_type[0], **inpt_type[1])
        erase_image_tensor_inpt = inpt
        fn = mocker.patch(
            "torchvision.prototype.transforms.functional.erase_image_tensor",
            return_value=mocker.MagicMock(spec=torch.Tensor),
        )
        if inpt_type[0] == PIL.Image.Image:
            erase_image_tensor_inpt = mocker.MagicMock(spec=torch.Tensor)

            # vfdev-5: I do not know how to patch pil_to_tensor if it is already imported
            # TODO: patch pil_to_tensor and run below checks for PIL.Image.Image inputs
            if p > 0.0:
                return

            mocker.patch(
                "torchvision.transforms.functional.pil_to_tensor",
                return_value=erase_image_tensor_inpt,
            )
            mocker.patch(
                "torchvision.transforms.functional.to_pil_image",
                return_value=mocker.MagicMock(spec=PIL.Image.Image),
            )

        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock()
        output = transform(inpt)
        print(inpt_type)
        assert isinstance(output, inpt_type[0])
        params = transform._get_params(inpt)
        if p > 0.0:
            fn.assert_called_once_with(erase_image_tensor_inpt, **params)
        else:
            fn.call_count == 0


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
        if inpt_type in (features.BoundingBox, str, int):
            fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, copy=transform.copy)


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
        if inpt_type in (features.BoundingBox, str, int):
            fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, copy=transform.copy)


class TestToPILImage:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, features.Image, np.ndarray, features.BoundingBox, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.functional.to_pil_image")

        inpt = mocker.MagicMock(spec=inpt_type)
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            transform = transforms.ToPILImage()
        transform(inpt)
        if inpt_type in (PIL.Image.Image, features.BoundingBox, str, int):
            fn.call_count == 0
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
            fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt)


class TestCompose:
    def test_assertions(self):
        with pytest.raises(TypeError, match="Argument transforms should be a sequence of callables"):
            transforms.Compose(123)

    @pytest.mark.parametrize(
        "trfms",
        [
            [transforms.Pad(2), transforms.RandomCrop(28)],
            [lambda x: 2.0 * x],
        ],
    )
    def test_ctor(self, trfms):
        c = transforms.Compose(trfms)
        inpt = torch.rand(1, 3, 32, 32)
        output = c(inpt)
        assert isinstance(output, torch.Tensor)


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
