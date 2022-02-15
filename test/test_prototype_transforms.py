import itertools

import PIL.Image
import pytest
import torch
from test_prototype_transforms_kernels import make_images, make_bounding_boxes, make_one_hot_labels
from torchvision.prototype import transforms, features
from torchvision.transforms.functional import to_pil_image


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


INPUT_CREATIONS_FNS = {
    features.Image: make_images,
    features.BoundingBox: make_bounding_boxes,
    features.OneHotLabel: make_one_hot_labels,
    torch.Tensor: make_vanilla_tensor_images,
    PIL.Image.Image: make_pil_images,
}


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
        dispatcher = transform._DISPATCHER
        if dispatcher is None:
            continue

        for type_ in dispatcher._kernels:
            try:
                inputs = INPUT_CREATIONS_FNS[type_]()
            except KeyError:
                continue

            transforms_with_inputs.append((transform, inputs))

    return parametrize(transforms_with_inputs)


class TestSmoke:
    @parametrize_from_transforms(
        transforms.RandomErasing(),
        transforms.HorizontalFlip(),
        transforms.Resize([16, 16]),
        transforms.CenterCrop([16, 16]),
        transforms.ConvertImageDtype(),
    )
    def test_common(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transform,
                [
                    *[features.Image.new_like(image, image.unsqueeze(0), dtype=torch.float) for image in make_images()],
                    *[
                        features.OneHotLabel.new_like(one_hot_label, one_hot_label.unsqueeze(0), dtype=torch.float)
                        for one_hot_label in make_one_hot_labels()
                    ],
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

    @parametrize(
        [
            (
                transform,
                itertools.chain.from_iterable(
                    fn(dtypes=[torch.uint8], extra_dims=[(4,)])
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
                    fn(color_spaces=["rgb"], dtypes=[torch.float32])
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
                transforms.ConvertColorSpace("grayscale"),
                itertools.chain(
                    make_images(),
                    make_vanilla_tensor_images(color_spaces=["rgb"]),
                    make_pil_images(color_spaces=["rgb"]),
                ),
            )
        ]
    )
    def test_convert_bounding_color_space(self, transform, input):
        transform(input)

    @parametrize(
        [
            (
                transforms.ConvertBoundingBoxFormat("xyxy", old_format="xywh"),
                itertools.chain(
                    make_bounding_boxes(),
                    make_vanilla_tensor_bounding_boxes(formats=["xywh"]),
                ),
            )
        ]
    )
    def test_convert_bounding_box_format(self, transform, input):
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
