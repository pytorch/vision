import itertools

import pytest
import torch
from common_utils import assert_equal
from test_prototype_transforms_functional import (
    make_images,
    make_bounding_boxes,
    make_one_hot_labels,
    make_segmentation_masks,
)
from torchvision.prototype import transforms, features
from torchvision.transforms.functional import to_pil_image, pil_to_tensor


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


def parametrize(transforms_with_inpts):
    return pytest.mark.parametrize(
        ("transform", "inpt"),
        [
            pytest.param(
                transform,
                inpt,
                id=f"{type(transform).__name__}-{type(inpt).__module__}.{type(inpt).__name__}-{idx}",
            )
            for transform, inpts in transforms_with_inpts
            for idx, inpt in enumerate(inpts)
        ],
    )


def parametrize_from_transforms(*transforms):
    transforms_with_inpts = []
    for transform in transforms:
        for creation_fn in [
            make_images,
            make_bounding_boxes,
            make_one_hot_labels,
            make_vanilla_tensor_images,
            make_pil_images,
            make_segmentation_masks,
        ]:
            inpts = list(creation_fn())
            # try:
            output = transform(inpts[0])
            # except TypeError:
            #     continue
            # else:
            #     if output is inpts[0]:
            #         continue

            transforms_with_inpts.append((transform, inpts))

    return parametrize(transforms_with_inpts)


class TestSmoke:
    @parametrize_from_transforms(
        # transforms.RandomErasing(p=1.0),
        transforms.Resize([16, 16]),
        transforms.CenterCrop([16, 16]),
        # transforms.ConvertImageDtype(),
        # transforms.RandomHorizontalFlip(),
        # transforms.Pad(5),
    )
    def test_common(self, transform, inpt):
        output = transform(inpt)
        assert type(output) == type(inpt)

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
    def test_mixup_cutmix(self, transform, inpt):
        transform(inpt)

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
    def test_auto_augment(self, transform, inpt):
        transform(inpt)

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
    def test_normalize(self, transform, inpt):
        transform(inpt)

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
    def test_random_resized_crop(self, transform, inpt):
        transform(inpt)

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
    def test_convert_image_color_space(self, transform, inpt):
        transform(inpt)


@pytest.mark.parametrize("p", [0.0, 1.0])
class TestRandomHorizontalFlip:
    def inpt_expected_image_tensor(self, p, dtype=torch.float32):
        inpt = torch.tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], dtype=dtype)
        expected = torch.tensor([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=dtype)

        return inpt, expected if p == 1 else inpt

    def test_simple_tensor(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(inpt)

        assert_equal(expected, actual)

    def test_pil_image(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p, dtype=torch.uint8)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(to_pil_image(inpt))

        assert_equal(expected, pil_to_tensor(actual))

    def test_features_image(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(features.Image(inpt))

        assert_equal(features.Image(expected), actual)

    def test_features_segmentation_mask(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(features.SegmentationMask(inpt))

        assert_equal(features.SegmentationMask(expected), actual)

    def test_features_bounding_box(self, p):
        inpt = features.BoundingBox([0, 0, 5, 5], format=features.BoundingBoxFormat.XYXY, image_size=(10, 10))
        transform = transforms.RandomHorizontalFlip(p=p)

        actual = transform(inpt)

        expected_image_tensor = torch.tensor([5, 0, 10, 5]) if p == 1.0 else inpt
        expected = features.BoundingBox.new_like(inpt, data=expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.image_size == expected.image_size


@pytest.mark.parametrize("p", [0.0, 1.0])
class TestRandomVerticalFlip:
    def inpt_expected_image_tensor(self, p, dtype=torch.float32):
        inpt = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=dtype)
        expected = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=dtype)

        return inpt, expected if p == 1 else inpt

    def test_simple_tensor(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(inpt)

        assert_equal(expected, actual)

    def test_pil_image(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p, dtype=torch.uint8)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(to_pil_image(inpt))

        assert_equal(expected, pil_to_tensor(actual))

    def test_features_image(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(features.Image(inpt))

        assert_equal(features.Image(expected), actual)

    def test_features_segmentation_mask(self, p):
        inpt, expected = self.inpt_expected_image_tensor(p)
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(features.SegmentationMask(inpt))

        assert_equal(features.SegmentationMask(expected), actual)

    def test_features_bounding_box(self, p):
        inpt = features.BoundingBox([0, 0, 5, 5], format=features.BoundingBoxFormat.XYXY, image_size=(10, 10))
        transform = transforms.RandomVerticalFlip(p=p)

        actual = transform(inpt)

        expected_image_tensor = torch.tensor([0, 5, 5, 10]) if p == 1.0 else inpt
        expected = features.BoundingBox.new_like(inpt, data=expected_image_tensor)
        assert_equal(expected, actual)
        assert actual.format == expected.format
        assert actual.image_size == expected.image_size
