import itertools
import pathlib
import random
import textwrap
import warnings

import numpy as np

import PIL.Image
import pytest
import torch
import torchvision.transforms.v2 as transforms

from common_utils import (
    assert_equal,
    assert_run_python_script,
    cpu_and_cuda,
    make_bounding_box,
    make_bounding_boxes,
    make_detection_mask,
    make_image,
    make_images,
    make_segmentation_mask,
    make_video,
    make_videos,
)
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import datapoints
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.utils import check_type, is_simple_tensor, query_chw


def make_vanilla_tensor_images(*args, **kwargs):
    for image in make_images(*args, **kwargs):
        if image.ndim > 3:
            continue
        yield image.data


def make_pil_images(*args, **kwargs):
    for image in make_vanilla_tensor_images(*args, **kwargs):
        yield to_pil_image(image)


def make_vanilla_tensor_bounding_boxes(*args, **kwargs):
    for bounding_boxes in make_bounding_boxes(*args, **kwargs):
        yield bounding_boxes.data


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


def auto_augment_adapter(transform, input, device):
    adapted_input = {}
    image_or_video_found = False
    for key, value in input.items():
        if isinstance(value, (datapoints.BoundingBoxes, datapoints.Mask)):
            # AA transforms don't support bounding boxes or masks
            continue
        elif check_type(value, (datapoints.Image, datapoints.Video, is_simple_tensor, PIL.Image.Image)):
            if image_or_video_found:
                # AA transforms only support a single image or video
                continue
            image_or_video_found = True
        adapted_input[key] = value
    return adapted_input


def linear_transformation_adapter(transform, input, device):
    flat_inputs = list(input.values())
    c, h, w = query_chw(
        [
            item
            for item, needs_transform in zip(flat_inputs, transforms.Transform()._needs_transform_list(flat_inputs))
            if needs_transform
        ]
    )
    num_elements = c * h * w
    transform.transformation_matrix = torch.randn((num_elements, num_elements), device=device)
    transform.mean_vector = torch.randn((num_elements,), device=device)
    return {key: value for key, value in input.items() if not isinstance(value, PIL.Image.Image)}


def normalize_adapter(transform, input, device):
    adapted_input = {}
    for key, value in input.items():
        if isinstance(value, PIL.Image.Image):
            # normalize doesn't support PIL images
            continue
        elif check_type(value, (datapoints.Image, datapoints.Video, is_simple_tensor)):
            # normalize doesn't support integer images
            value = F.to_dtype(value, torch.float32, scale=True)
        adapted_input[key] = value
    return adapted_input


class TestSmoke:
    @pytest.mark.parametrize(
        ("transform", "adapter"),
        [
            (transforms.RandomErasing(p=1.0), None),
            (transforms.AugMix(), auto_augment_adapter),
            (transforms.AutoAugment(), auto_augment_adapter),
            (transforms.RandAugment(), auto_augment_adapter),
            (transforms.TrivialAugmentWide(), auto_augment_adapter),
            (transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.15), None),
            (transforms.Grayscale(), None),
            (transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0), None),
            (transforms.RandomAutocontrast(p=1.0), None),
            (transforms.RandomEqualize(p=1.0), None),
            (transforms.RandomGrayscale(p=1.0), None),
            (transforms.RandomInvert(p=1.0), None),
            (transforms.RandomPermuteChannels(), None),
            (transforms.RandomPhotometricDistort(p=1.0), None),
            (transforms.RandomPosterize(bits=4, p=1.0), None),
            (transforms.RandomSolarize(threshold=0.5, p=1.0), None),
            (transforms.CenterCrop([16, 16]), None),
            (transforms.ElasticTransform(sigma=1.0), None),
            (transforms.Pad(4), None),
            (transforms.RandomAffine(degrees=30.0), None),
            (transforms.RandomCrop([16, 16], pad_if_needed=True), None),
            (transforms.RandomHorizontalFlip(p=1.0), None),
            (transforms.RandomPerspective(p=1.0), None),
            (transforms.RandomResize(min_size=10, max_size=20, antialias=True), None),
            (transforms.RandomResizedCrop([16, 16], antialias=True), None),
            (transforms.RandomRotation(degrees=30), None),
            (transforms.RandomShortestSize(min_size=10, antialias=True), None),
            (transforms.RandomVerticalFlip(p=1.0), None),
            (transforms.RandomZoomOut(p=1.0), None),
            (transforms.Resize([16, 16], antialias=True), None),
            (transforms.ScaleJitter((16, 16), scale_range=(0.8, 1.2), antialias=True), None),
            (transforms.ClampBoundingBoxes(), None),
            (transforms.ConvertBoundingBoxFormat(datapoints.BoundingBoxFormat.CXCYWH), None),
            (transforms.ConvertImageDtype(), None),
            (transforms.GaussianBlur(kernel_size=3), None),
            (
                transforms.LinearTransformation(
                    # These are just dummy values that will be filled by the adapter. We can't define them upfront,
                    # because for we neither know the spatial size nor the device at this point
                    transformation_matrix=torch.empty((1, 1)),
                    mean_vector=torch.empty((1,)),
                ),
                linear_transformation_adapter,
            ),
            (transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), normalize_adapter),
            (transforms.ToDtype(torch.float64), None),
            (transforms.UniformTemporalSubsample(num_samples=2), None),
        ],
        ids=lambda transform: type(transform).__name__,
    )
    @pytest.mark.parametrize("container_type", [dict, list, tuple])
    @pytest.mark.parametrize(
        "image_or_video",
        [
            make_image(),
            make_video(),
            next(make_pil_images(color_spaces=["RGB"])),
            next(make_vanilla_tensor_images()),
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_common(self, transform, adapter, container_type, image_or_video, device):
        canvas_size = F.get_size(image_or_video)
        input = dict(
            image_or_video=image_or_video,
            image_datapoint=make_image(size=canvas_size),
            video_datapoint=make_video(size=canvas_size),
            image_pil=next(make_pil_images(sizes=[canvas_size], color_spaces=["RGB"])),
            bounding_boxes_xyxy=make_bounding_box(
                format=datapoints.BoundingBoxFormat.XYXY, canvas_size=canvas_size, batch_dims=(3,)
            ),
            bounding_boxes_xywh=make_bounding_box(
                format=datapoints.BoundingBoxFormat.XYWH, canvas_size=canvas_size, batch_dims=(4,)
            ),
            bounding_boxes_cxcywh=make_bounding_box(
                format=datapoints.BoundingBoxFormat.CXCYWH, canvas_size=canvas_size, batch_dims=(5,)
            ),
            bounding_boxes_degenerate_xyxy=datapoints.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [2, 0, 1, 1],  # x1 > x2, y1 < y2
                    [0, 2, 1, 1],  # x1 < x2, y1 > y2
                    [2, 2, 1, 1],  # x1 > x2, y1 > y2
                ],
                format=datapoints.BoundingBoxFormat.XYXY,
                canvas_size=canvas_size,
            ),
            bounding_boxes_degenerate_xywh=datapoints.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=datapoints.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size,
            ),
            bounding_boxes_degenerate_cxcywh=datapoints.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=datapoints.BoundingBoxFormat.CXCYWH,
                canvas_size=canvas_size,
            ),
            detection_mask=make_detection_mask(size=canvas_size),
            segmentation_mask=make_segmentation_mask(size=canvas_size),
            int=0,
            float=0.0,
            bool=True,
            none=None,
            str="str",
            path=pathlib.Path.cwd(),
            object=object(),
            tensor=torch.empty(5),
            array=np.empty(5),
        )
        if adapter is not None:
            input = adapter(transform, input, device)

        if container_type in {tuple, list}:
            input = container_type(input.values())

        input_flat, input_spec = tree_flatten(input)
        input_flat = [item.to(device) if isinstance(item, torch.Tensor) else item for item in input_flat]
        input = tree_unflatten(input_flat, input_spec)

        torch.manual_seed(0)
        output = transform(input)
        output_flat, output_spec = tree_flatten(output)

        assert output_spec == input_spec

        for output_item, input_item, should_be_transformed in zip(
            output_flat, input_flat, transforms.Transform()._needs_transform_list(input_flat)
        ):
            if should_be_transformed:
                assert type(output_item) is type(input_item)
            else:
                assert output_item is input_item

            if isinstance(input_item, datapoints.BoundingBoxes) and not isinstance(
                transform, transforms.ConvertBoundingBoxFormat
            ):
                assert output_item.format == input_item.format

        # Enforce that the transform does not turn a degenerate box marked by RandomIoUCrop (or any other future
        # transform that does this), back into a valid one.
        # TODO: we should test that against all degenerate boxes above
        for format in list(datapoints.BoundingBoxFormat):
            sample = dict(
                boxes=datapoints.BoundingBoxes([[0, 0, 0, 0]], format=format, canvas_size=(224, 244)),
                labels=torch.tensor([3]),
            )
            assert transforms.SanitizeBoundingBoxes()(sample)["boxes"].shape == (0, 4)

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
                        **(dict(num_frames=[3]) if fn is make_videos else dict()),
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
                transforms.RandomResizedCrop([16, 16], antialias=True),
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

        fn = mocker.patch("torchvision.transforms.v2.functional.pad")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        _ = transform(inpt)

        fill = transforms._utils._convert_fill_arg(fill)
        if isinstance(padding, tuple):
            padding = list(padding)
        fn.assert_called_once_with(inpt, padding=padding, fill=fill, padding_mode=padding_mode)

    @pytest.mark.parametrize("fill", [12, {datapoints.Image: 12, datapoints.Mask: 34}])
    def test__transform_image_mask(self, fill, mocker):
        transform = transforms.Pad(1, fill=fill, padding_mode="constant")

        fn = mocker.patch("torchvision.transforms.v2.functional.pad")
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
    def test__get_params(self, fill, side_range):
        transform = transforms.RandomZoomOut(fill=fill, side_range=side_range)

        h, w = size = (24, 32)
        image = make_image(size)

        params = transform._get_params([image])

        assert len(params["padding"]) == 4
        assert 0 <= params["padding"][0] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][1] <= (side_range[1] - 1) * h
        assert 0 <= params["padding"][2] <= (side_range[1] - 1) * w
        assert 0 <= params["padding"][3] <= (side_range[1] - 1) * h

    @pytest.mark.parametrize("fill", [0, [1, 2, 3], (2, 3, 4)])
    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    def test__transform(self, fill, side_range, mocker):
        inpt = make_image((24, 32))

        transform = transforms.RandomZoomOut(fill=fill, side_range=side_range, p=1)

        fn = mocker.patch("torchvision.transforms.v2.functional.pad")
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

        fn = mocker.patch("torchvision.transforms.v2.functional.pad")
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
    def test__get_params(self, padding, pad_if_needed, size):
        h, w = size = (24, 32)
        image = make_image(size)

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

        h, w = size = (32, 32)
        inpt = make_image(size)

        if isinstance(padding, int):
            new_size = (h + padding, w + padding)
        elif isinstance(padding, list):
            new_size = (h + sum(padding[0::2]), w + sum(padding[1::2]))
        else:
            new_size = size
        expected = make_image(new_size)
        _ = mocker.patch("torchvision.transforms.v2.functional.pad", return_value=expected)
        fn_crop = mocker.patch("torchvision.transforms.v2.functional.crop")

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

        fn = mocker.patch("torchvision.transforms.v2.functional.gaussian_blur")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.canvas_size = (24, 32)

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

        fn = mocker.patch(f"torchvision.transforms.v2.functional.{func_op_name}")
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

    def test__get_params(self):
        dscale = 0.5
        transform = transforms.RandomPerspective(dscale)

        image = make_image((24, 32))

        params = transform._get_params([image])

        assert "coefficients" in params
        assert len(params["coefficients"]) == 8

    @pytest.mark.parametrize("distortion_scale", [0.1, 0.7])
    def test__transform(self, distortion_scale, mocker):
        interpolation = InterpolationMode.BILINEAR
        fill = 12
        transform = transforms.RandomPerspective(distortion_scale, fill=fill, interpolation=interpolation)

        fn = mocker.patch("torchvision.transforms.v2.functional.perspective")

        inpt = make_image((24, 32))

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

    def test__get_params(self):
        alpha = 2.0
        sigma = 3.0
        transform = transforms.ElasticTransform(alpha, sigma)

        h, w = size = (24, 32)
        image = make_image(size)

        params = transform._get_params([image])

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

        fn = mocker.patch("torchvision.transforms.v2.functional.elastic")
        inpt = mocker.MagicMock(spec=datapoints.Image)
        inpt.num_channels = 3
        inpt.canvas_size = (24, 32)

        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock()
        _ = transform(inpt)
        params = transform._get_params([inpt])
        fill = transforms._utils._convert_fill_arg(fill)
        fn.assert_called_once_with(inpt, **params, fill=fill, interpolation=interpolation)


class TestRandomErasing:
    def test_assertions(self):
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

        image = make_image((24, 32))

        transform = transforms.RandomErasing(value=[1, 2, 3, 4])

        with pytest.raises(ValueError, match="If value is a sequence, it should have either a single value"):
            transform._get_params([image])

    @pytest.mark.parametrize("value", [5.0, [1, 2, 3], "random"])
    def test__get_params(self, value):
        image = make_image((24, 32))
        num_channels, height, width = F.get_dimensions(image)

        transform = transforms.RandomErasing(value=value)
        params = transform._get_params([image])

        v = params["v"]
        h, w = params["h"], params["w"]
        i, j = params["i"], params["j"]
        assert isinstance(v, torch.Tensor)
        if value == "random":
            assert v.shape == (num_channels, h, w)
        elif isinstance(value, (int, float)):
            assert v.shape == (1, 1, 1)
        elif isinstance(value, (list, tuple)):
            assert v.shape == (num_channels, 1, 1)

        assert 0 <= i <= height - h
        assert 0 <= j <= width - w

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
            "torchvision.transforms.v2._augment.RandomErasing._get_params",
            return_value=dict(i=i_sentinel, j=j_sentinel, h=h_sentinel, w=w_sentinel, v=v_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.transforms.v2._augment.F.erase")
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
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBoxes, str, int],
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
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBoxes, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch(
            "torchvision.transforms.v2.functional.to_image_tensor",
            return_value=torch.rand(1, 3, 8, 8),
        )

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImageTensor()
        transform(inpt)
        if inpt_type in (datapoints.BoundingBoxes, datapoints.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt)


class TestToImagePIL:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBoxes, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.v2.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToImagePIL()
        transform(inpt)
        if inpt_type in (datapoints.BoundingBoxes, PIL.Image.Image, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToPILImage:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBoxes, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.v2.functional.to_image_pil")

        inpt = mocker.MagicMock(spec=inpt_type)
        transform = transforms.ToPILImage()
        transform(inpt)
        if inpt_type in (PIL.Image.Image, datapoints.BoundingBoxes, str, int):
            assert fn.call_count == 0
        else:
            fn.assert_called_once_with(inpt, mode=transform.mode)


class TestToTensor:
    @pytest.mark.parametrize(
        "inpt_type",
        [torch.Tensor, PIL.Image.Image, datapoints.Image, np.ndarray, datapoints.BoundingBoxes, str, int],
    )
    def test__transform(self, inpt_type, mocker):
        fn = mocker.patch("torchvision.transforms.functional.to_tensor")

        inpt = mocker.MagicMock(spec=inpt_type)
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            transform = transforms.ToTensor()
        transform(inpt)
        if inpt_type in (datapoints.Image, torch.Tensor, datapoints.BoundingBoxes, str, int):
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
        with pytest.raises(ValueError, match="Length of p doesn't match the number of transforms"):
            transforms.RandomChoice([transforms.Pad(2), transforms.RandomCrop(28)], p=[1])


class TestRandomIoUCrop:
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("options", [[0.5, 0.9], [2.0]])
    def test__get_params(self, device, options):
        orig_h, orig_w = size = (24, 32)
        image = make_image(size)
        bboxes = datapoints.BoundingBoxes(
            torch.tensor([[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]),
            format="XYXY",
            canvas_size=size,
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
        bboxes = datapoints.BoundingBoxes(torch.tensor([[1, 1, 2, 2]]), format="XYXY", canvas_size=(4, 4))
        label = torch.tensor([1])
        sample = [image, bboxes, label]
        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock(return_value={})
        output = transform(sample)
        torch.testing.assert_close(output, sample)

    def test_forward_assertion(self):
        transform = transforms.RandomIoUCrop()
        with pytest.raises(
            TypeError,
            match="requires input sample to contain tensor or PIL images and bounding boxes",
        ):
            transform(torch.tensor(0))

    def test__transform(self, mocker):
        transform = transforms.RandomIoUCrop()

        size = (32, 24)
        image = make_image(size)
        bboxes = make_bounding_box(format="XYXY", canvas_size=size, batch_dims=(6,))
        masks = make_detection_mask(size, num_objects=6)

        sample = [image, bboxes, masks]

        fn = mocker.patch("torchvision.transforms.v2.functional.crop", side_effect=lambda x, **params: x)
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

        # check number of bboxes vs number of labels:
        output_bboxes = output[1]
        assert isinstance(output_bboxes, datapoints.BoundingBoxes)
        assert (output_bboxes[~is_within_crop_area] == 0).all()

        output_masks = output[2]
        assert isinstance(output_masks, datapoints.Mask)


class TestScaleJitter:
    def test__get_params(self):
        canvas_size = (24, 32)
        target_size = (16, 12)
        scale_range = (0.5, 1.5)

        transform = transforms.ScaleJitter(target_size=target_size, scale_range=scale_range)

        sample = make_image(canvas_size)

        n_samples = 5
        for _ in range(n_samples):

            params = transform._get_params([sample])

            assert "size" in params
            size = params["size"]

            assert isinstance(size, tuple) and len(size) == 2
            height, width = size

            r_min = min(target_size[1] / canvas_size[0], target_size[0] / canvas_size[1]) * scale_range[0]
            r_max = min(target_size[1] / canvas_size[0], target_size[0] / canvas_size[1]) * scale_range[1]

            assert int(canvas_size[0] * r_min) <= height <= int(canvas_size[0] * r_max)
            assert int(canvas_size[1] * r_min) <= width <= int(canvas_size[1] * r_max)

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock(spec=InterpolationMode)
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.ScaleJitter(
            target_size=(16, 12), interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.transforms.v2._geometry.ScaleJitter._get_params", return_value=dict(size=size_sentinel)
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.transforms.v2._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(
            inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


class TestRandomShortestSize:
    @pytest.mark.parametrize("min_size,max_size", [([5, 9], 20), ([5, 9], None)])
    def test__get_params(self, min_size, max_size):
        canvas_size = (3, 10)

        transform = transforms.RandomShortestSize(min_size=min_size, max_size=max_size, antialias=True)

        sample = make_image(canvas_size)
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
        interpolation_sentinel = mocker.MagicMock(spec=InterpolationMode)
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.RandomShortestSize(
            min_size=[3, 5, 7], max_size=12, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.transforms.v2._geometry.RandomShortestSize._get_params",
            return_value=dict(size=size_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock = mocker.patch("torchvision.transforms.v2._geometry.F.resize")
        transform(inpt_sentinel)

        mock.assert_called_once_with(
            inpt_sentinel, size=size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


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


class TestRandomResize:
    def test__get_params(self):
        min_size = 3
        max_size = 6

        transform = transforms.RandomResize(min_size=min_size, max_size=max_size, antialias=True)

        for _ in range(10):
            params = transform._get_params([])

            assert isinstance(params["size"], list) and len(params["size"]) == 1
            size = params["size"][0]

            assert min_size <= size < max_size

    def test__transform(self, mocker):
        interpolation_sentinel = mocker.MagicMock(spec=InterpolationMode)
        antialias_sentinel = mocker.MagicMock()

        transform = transforms.RandomResize(
            min_size=-1, max_size=-1, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )
        transform._transformed_types = (mocker.MagicMock,)

        size_sentinel = mocker.MagicMock()
        mocker.patch(
            "torchvision.transforms.v2._geometry.RandomResize._get_params",
            return_value=dict(size=size_sentinel),
        )

        inpt_sentinel = mocker.MagicMock()

        mock_resize = mocker.patch("torchvision.transforms.v2._geometry.F.resize")
        transform(inpt_sentinel)

        mock_resize.assert_called_with(
            inpt_sentinel, size_sentinel, interpolation=interpolation_sentinel, antialias=antialias_sentinel
        )


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


# TODO: remove this test in 0.17 when the default of antialias changes to True
def test_antialias_warning():
    pil_img = PIL.Image.new("RGB", size=(10, 10), color=127)
    tensor_img = torch.randint(0, 256, size=(3, 10, 10), dtype=torch.uint8)
    tensor_video = torch.randint(0, 256, size=(2, 3, 10, 10), dtype=torch.uint8)

    match = "The default value of the antialias parameter"
    with pytest.warns(UserWarning, match=match):
        transforms.RandomResizedCrop((20, 20))(tensor_img)
    with pytest.warns(UserWarning, match=match):
        transforms.ScaleJitter((20, 20))(tensor_img)
    with pytest.warns(UserWarning, match=match):
        transforms.RandomShortestSize((20, 20))(tensor_img)
    with pytest.warns(UserWarning, match=match):
        transforms.RandomResize(10, 20)(tensor_img)

    with pytest.warns(UserWarning, match=match):
        F.resized_crop(datapoints.Image(tensor_img), 0, 0, 10, 10, (20, 20))

    with pytest.warns(UserWarning, match=match):
        F.resize(datapoints.Video(tensor_video), (20, 20))
    with pytest.warns(UserWarning, match=match):
        F.resized_crop(datapoints.Video(tensor_video), 0, 0, 10, 10, (20, 20))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transforms.RandomResizedCrop((20, 20))(pil_img)
        transforms.ScaleJitter((20, 20))(pil_img)
        transforms.RandomShortestSize((20, 20))(pil_img)
        transforms.RandomResize(10, 20)(pil_img)

        transforms.RandomResizedCrop((20, 20), antialias=True)(tensor_img)
        transforms.ScaleJitter((20, 20), antialias=True)(tensor_img)
        transforms.RandomShortestSize((20, 20), antialias=True)(tensor_img)
        transforms.RandomResize(10, 20, antialias=True)(tensor_img)

        F.resized_crop(datapoints.Image(tensor_img), 0, 0, 10, 10, (20, 20), antialias=True)
        F.resized_crop(datapoints.Video(tensor_video), 0, 0, 10, 10, (20, 20), antialias=True)


@pytest.mark.parametrize("image_type", (PIL.Image, torch.Tensor, datapoints.Image))
@pytest.mark.parametrize("label_type", (torch.Tensor, int))
@pytest.mark.parametrize("dataset_return_type", (dict, tuple))
@pytest.mark.parametrize("to_tensor", (transforms.ToTensor, transforms.ToImageTensor))
def test_classif_preset(image_type, label_type, dataset_return_type, to_tensor):

    image = datapoints.Image(torch.randint(0, 256, size=(1, 3, 250, 250), dtype=torch.uint8))
    if image_type is PIL.Image:
        image = to_pil_image(image[0])
    elif image_type is torch.Tensor:
        image = image.as_subclass(torch.Tensor)
        assert is_simple_tensor(image)

    label = 1 if label_type is int else torch.tensor([1])

    if dataset_return_type is dict:
        sample = {
            "image": image,
            "label": label,
        }
    else:
        sample = image, label

    if to_tensor is transforms.ToTensor:
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            to_tensor = to_tensor()
    else:
        to_tensor = to_tensor()

    t = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.AugMix(),
            transforms.AutoAugment(),
            to_tensor,
            # TODO: ConvertImageDtype is a pass-through on PIL images, is that
            # intended?  This results in a failure if we convert to tensor after
            # it, because the image would still be uint8 which make Normalize
            # fail.
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            transforms.RandomErasing(p=1),
        ]
    )

    out = t(sample)

    assert type(out) == type(sample)

    if dataset_return_type is tuple:
        out_image, out_label = out
    else:
        assert out.keys() == sample.keys()
        out_image, out_label = out.values()

    assert out_image.shape[-2:] == (224, 224)
    assert out_label == label


@pytest.mark.parametrize("image_type", (PIL.Image, torch.Tensor, datapoints.Image))
@pytest.mark.parametrize("data_augmentation", ("hflip", "lsj", "multiscale", "ssd", "ssdlite"))
@pytest.mark.parametrize("to_tensor", (transforms.ToTensor, transforms.ToImageTensor))
@pytest.mark.parametrize("sanitize", (True, False))
def test_detection_preset(image_type, data_augmentation, to_tensor, sanitize):
    torch.manual_seed(0)

    if to_tensor is transforms.ToTensor:
        with pytest.warns(UserWarning, match="deprecated and will be removed"):
            to_tensor = to_tensor()
    else:
        to_tensor = to_tensor()

    if data_augmentation == "hflip":
        t = [
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "lsj":
        t = [
            transforms.ScaleJitter(target_size=(1024, 1024), antialias=True),
            # Note: replaced FixedSizeCrop with RandomCrop, becuase we're
            # leaving FixedSizeCrop in prototype for now, and it expects Label
            # classes which we won't release yet.
            # transforms.FixedSizeCrop(
            #     size=(1024, 1024), fill=defaultdict(lambda: (123.0, 117.0, 104.0), {datapoints.Mask: 0})
            # ),
            transforms.RandomCrop((1024, 1024), pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "multiscale":
        t = [
            transforms.RandomShortestSize(
                min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssd":
        t = [
            transforms.RandomPhotometricDistort(p=1),
            transforms.RandomZoomOut(fill={"others": (123.0, 117.0, 104.0), datapoints.Mask: 0}, p=1),
            transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssdlite":
        t = [
            transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=1),
            to_tensor,
            transforms.ConvertImageDtype(torch.float),
        ]
    if sanitize:
        t += [transforms.SanitizeBoundingBoxes()]
    t = transforms.Compose(t)

    num_boxes = 5
    H = W = 250

    image = datapoints.Image(torch.randint(0, 256, size=(1, 3, H, W), dtype=torch.uint8))
    if image_type is PIL.Image:
        image = to_pil_image(image[0])
    elif image_type is torch.Tensor:
        image = image.as_subclass(torch.Tensor)
        assert is_simple_tensor(image)

    label = torch.randint(0, 10, size=(num_boxes,))

    boxes = torch.randint(0, min(H, W) // 2, size=(num_boxes, 4))
    boxes[:, 2:] += boxes[:, :2]
    boxes = boxes.clamp(min=0, max=min(H, W))
    boxes = datapoints.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))

    masks = datapoints.Mask(torch.randint(0, 2, size=(num_boxes, H, W), dtype=torch.uint8))

    sample = {
        "image": image,
        "label": label,
        "boxes": boxes,
        "masks": masks,
    }

    out = t(sample)

    if isinstance(to_tensor, transforms.ToTensor) and image_type is not datapoints.Image:
        assert is_simple_tensor(out["image"])
    else:
        assert isinstance(out["image"], datapoints.Image)
    assert isinstance(out["label"], type(sample["label"]))

    num_boxes_expected = {
        # ssd and ssdlite contain RandomIoUCrop which may "remove" some bbox. It
        # doesn't remove them strictly speaking, it just marks some boxes as
        # degenerate and those boxes will be later removed by
        # SanitizeBoundingBoxes(), which we add to the pipelines if the sanitize
        # param is True.
        # Note that the values below are probably specific to the random seed
        # set above (which is fine).
        (True, "ssd"): 5,
        (True, "ssdlite"): 4,
    }.get((sanitize, data_augmentation), num_boxes)

    assert out["boxes"].shape[0] == out["masks"].shape[0] == out["label"].shape[0] == num_boxes_expected


@pytest.mark.parametrize("min_size", (1, 10))
@pytest.mark.parametrize("labels_getter", ("default", lambda inputs: inputs["labels"], None, lambda inputs: None))
@pytest.mark.parametrize("sample_type", (tuple, dict))
def test_sanitize_bounding_boxes(min_size, labels_getter, sample_type):

    if sample_type is tuple and not isinstance(labels_getter, str):
        # The "lambda inputs: inputs["labels"]" labels_getter used in this test
        # doesn't work if the input is a tuple.
        return

    H, W = 256, 128

    boxes_and_validity = [
        ([0, 1, 10, 1], False),  # Y1 == Y2
        ([0, 1, 0, 20], False),  # X1 == X2
        ([0, 0, min_size - 1, 10], False),  # H < min_size
        ([0, 0, 10, min_size - 1], False),  # W < min_size
        ([0, 0, 10, H + 1], False),  # Y2 > H
        ([0, 0, W + 1, 10], False),  # X2 > W
        ([-1, 1, 10, 20], False),  # any < 0
        ([0, 0, -1, 20], False),  # any < 0
        ([0, 0, -10, -1], False),  # any < 0
        ([0, 0, min_size, 10], True),  # H < min_size
        ([0, 0, 10, min_size], True),  # W < min_size
        ([0, 0, W, H], True),  # TODO: Is that actually OK?? Should it be -1?
        ([1, 1, 30, 20], True),
        ([0, 0, 10, 10], True),
        ([1, 1, 30, 20], True),
    ]

    random.shuffle(boxes_and_validity)  # For test robustness: mix order of wrong and correct cases
    boxes, is_valid_mask = zip(*boxes_and_validity)
    valid_indices = [i for (i, is_valid) in enumerate(is_valid_mask) if is_valid]

    boxes = torch.tensor(boxes)
    labels = torch.arange(boxes.shape[0])

    boxes = datapoints.BoundingBoxes(
        boxes,
        format=datapoints.BoundingBoxFormat.XYXY,
        canvas_size=(H, W),
    )

    masks = datapoints.Mask(torch.randint(0, 2, size=(boxes.shape[0], H, W)))
    whatever = torch.rand(10)
    input_img = torch.randint(0, 256, size=(1, 3, H, W), dtype=torch.uint8)
    sample = {
        "image": input_img,
        "labels": labels,
        "boxes": boxes,
        "whatever": whatever,
        "None": None,
        "masks": masks,
    }

    if sample_type is tuple:
        img = sample.pop("image")
        sample = (img, sample)

    out = transforms.SanitizeBoundingBoxes(min_size=min_size, labels_getter=labels_getter)(sample)

    if sample_type is tuple:
        out_image = out[0]
        out_labels = out[1]["labels"]
        out_boxes = out[1]["boxes"]
        out_masks = out[1]["masks"]
        out_whatever = out[1]["whatever"]
    else:
        out_image = out["image"]
        out_labels = out["labels"]
        out_boxes = out["boxes"]
        out_masks = out["masks"]
        out_whatever = out["whatever"]

    assert out_image is input_img
    assert out_whatever is whatever

    assert isinstance(out_boxes, datapoints.BoundingBoxes)
    assert isinstance(out_masks, datapoints.Mask)

    if labels_getter is None or (callable(labels_getter) and labels_getter({"labels": "blah"}) is None):
        assert out_labels is labels
    else:
        assert isinstance(out_labels, torch.Tensor)
        assert out_boxes.shape[0] == out_labels.shape[0] == out_masks.shape[0]
        # This works because we conveniently set labels to arange(num_boxes)
        assert out_labels.tolist() == valid_indices


def test_sanitize_bounding_boxes_errors():

    good_bbox = datapoints.BoundingBoxes(
        [[0, 0, 10, 10]],
        format=datapoints.BoundingBoxFormat.XYXY,
        canvas_size=(20, 20),
    )

    with pytest.raises(ValueError, match="min_size must be >= 1"):
        transforms.SanitizeBoundingBoxes(min_size=0)
    with pytest.raises(ValueError, match="labels_getter should either be 'default'"):
        transforms.SanitizeBoundingBoxes(labels_getter=12)

    with pytest.raises(ValueError, match="Could not infer where the labels are"):
        bad_labels_key = {"bbox": good_bbox, "BAD_KEY": torch.arange(good_bbox.shape[0])}
        transforms.SanitizeBoundingBoxes()(bad_labels_key)

    with pytest.raises(ValueError, match="must be a tensor"):
        not_a_tensor = {"bbox": good_bbox, "labels": torch.arange(good_bbox.shape[0]).tolist()}
        transforms.SanitizeBoundingBoxes()(not_a_tensor)

    with pytest.raises(ValueError, match="Number of boxes"):
        different_sizes = {"bbox": good_bbox, "labels": torch.arange(good_bbox.shape[0] + 3)}
        transforms.SanitizeBoundingBoxes()(different_sizes)


@pytest.mark.parametrize(
    "import_statement",
    (
        "from torchvision.transforms import v2",
        "import torchvision.transforms.v2",
        "from torchvision.transforms.v2 import Resize",
        "import torchvision.transforms.v2.functional",
        "from torchvision.transforms.v2.functional import resize",
        "from torchvision import datapoints",
        "from torchvision.datapoints import Image",
        "from torchvision.datasets import wrap_dataset_for_transforms_v2",
    ),
)
@pytest.mark.parametrize("call_disable_warning", (True, False))
def test_warnings_v2_namespaces(import_statement, call_disable_warning):
    if call_disable_warning:
        source = f"""
        import warnings
        import torchvision
        torchvision.disable_beta_transforms_warning()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            {import_statement}
        """
    else:
        source = f"""
        import pytest
        with pytest.warns(UserWarning, match="v2 namespaces are still Beta"):
            {import_statement}
        """
    assert_run_python_script(textwrap.dedent(source))


def test_no_warnings_v1_namespace():
    source = """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import torchvision.transforms
        from torchvision import transforms
        import torchvision.transforms.functional
        from torchvision.transforms import Resize
        from torchvision.transforms.functional import resize
        from torchvision import datasets
        from torchvision.datasets import ImageNet
    """
    assert_run_python_script(textwrap.dedent(source))


class TestLambda:
    inputs = pytest.mark.parametrize("input", [object(), torch.empty(()), np.empty(()), "string", 1, 0.0])

    @inputs
    def test_default(self, input):
        was_applied = False

        def was_applied_fn(input):
            nonlocal was_applied
            was_applied = True
            return input

        transform = transforms.Lambda(was_applied_fn)

        transform(input)

        assert was_applied

    @inputs
    def test_with_types(self, input):
        was_applied = False

        def was_applied_fn(input):
            nonlocal was_applied
            was_applied = True
            return input

        types = (torch.Tensor, np.ndarray)
        transform = transforms.Lambda(was_applied_fn, *types)

        transform(input)

        assert was_applied is isinstance(input, types)
