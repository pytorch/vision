import math
import os
import random
import re
import sys
from functools import partial

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
import torchvision.transforms._functional_tensor as F_t
import torchvision.transforms.functional as F
from PIL import Image
from torch._utils_internal import get_file_path_2

try:
    import accimage
except ImportError:
    accimage = None

try:
    from scipy import stats
except ImportError:
    stats = None

from common_utils import assert_equal, cycle_over, float_dtypes, int_dtypes


GRACE_HOPPER = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
)


def _get_grayscale_test_image(img, fill=None):
    img = img.convert("L")
    fill = (fill[0],) if isinstance(fill, tuple) else fill
    return img, fill


class TestConvertImageDtype:
    @pytest.mark.parametrize("input_dtype, output_dtype", cycle_over(float_dtypes()))
    def test_float_to_float(self, input_dtype, output_dtype):
        input_image = torch.tensor((0.0, 1.0), dtype=input_dtype)
        transform = transforms.ConvertImageDtype(output_dtype)
        transform_script = torch.jit.script(F.convert_image_dtype)

        output_image = transform(input_image)
        output_image_script = transform_script(input_image, output_dtype)

        torch.testing.assert_close(output_image_script, output_image, rtol=0.0, atol=1e-6)

        actual_min, actual_max = output_image.tolist()
        desired_min, desired_max = 0.0, 1.0

        assert abs(actual_min - desired_min) < 1e-7
        assert abs(actual_max - desired_max) < 1e-7

    @pytest.mark.parametrize("input_dtype", float_dtypes())
    @pytest.mark.parametrize("output_dtype", int_dtypes())
    def test_float_to_int(self, input_dtype, output_dtype):
        input_image = torch.tensor((0.0, 1.0), dtype=input_dtype)
        transform = transforms.ConvertImageDtype(output_dtype)
        transform_script = torch.jit.script(F.convert_image_dtype)

        if (input_dtype == torch.float32 and output_dtype in (torch.int32, torch.int64)) or (
            input_dtype == torch.float64 and output_dtype == torch.int64
        ):
            with pytest.raises(RuntimeError):
                transform(input_image)
        else:
            output_image = transform(input_image)
            output_image_script = transform_script(input_image, output_dtype)

            torch.testing.assert_close(output_image_script, output_image, rtol=0.0, atol=1e-6)

            actual_min, actual_max = output_image.tolist()
            desired_min, desired_max = 0, torch.iinfo(output_dtype).max

            assert actual_min == desired_min
            assert actual_max == desired_max

    @pytest.mark.parametrize("input_dtype", int_dtypes())
    @pytest.mark.parametrize("output_dtype", float_dtypes())
    def test_int_to_float(self, input_dtype, output_dtype):
        input_image = torch.tensor((0, torch.iinfo(input_dtype).max), dtype=input_dtype)
        transform = transforms.ConvertImageDtype(output_dtype)
        transform_script = torch.jit.script(F.convert_image_dtype)

        output_image = transform(input_image)
        output_image_script = transform_script(input_image, output_dtype)

        torch.testing.assert_close(output_image_script, output_image, rtol=0.0, atol=1e-6)

        actual_min, actual_max = output_image.tolist()
        desired_min, desired_max = 0.0, 1.0

        assert abs(actual_min - desired_min) < 1e-7
        assert actual_min >= desired_min
        assert abs(actual_max - desired_max) < 1e-7
        assert actual_max <= desired_max

    @pytest.mark.parametrize("input_dtype, output_dtype", cycle_over(int_dtypes()))
    def test_dtype_int_to_int(self, input_dtype, output_dtype):
        input_max = torch.iinfo(input_dtype).max
        input_image = torch.tensor((0, input_max), dtype=input_dtype)
        output_max = torch.iinfo(output_dtype).max

        transform = transforms.ConvertImageDtype(output_dtype)
        transform_script = torch.jit.script(F.convert_image_dtype)

        output_image = transform(input_image)
        output_image_script = transform_script(input_image, output_dtype)

        torch.testing.assert_close(
            output_image_script,
            output_image,
            rtol=0.0,
            atol=1e-6,
            msg=f"{output_image_script} vs {output_image}",
        )

        actual_min, actual_max = output_image.tolist()
        desired_min, desired_max = 0, output_max

        # see https://github.com/pytorch/vision/pull/2078#issuecomment-641036236 for details
        if input_max >= output_max:
            error_term = 0
        else:
            error_term = 1 - (torch.iinfo(output_dtype).max + 1) // (torch.iinfo(input_dtype).max + 1)

        assert actual_min == desired_min
        assert actual_max == (desired_max + error_term)

    @pytest.mark.parametrize("input_dtype, output_dtype", cycle_over(int_dtypes()))
    def test_int_to_int_consistency(self, input_dtype, output_dtype):
        input_max = torch.iinfo(input_dtype).max
        input_image = torch.tensor((0, input_max), dtype=input_dtype)

        output_max = torch.iinfo(output_dtype).max
        if output_max <= input_max:
            return

        transform = transforms.ConvertImageDtype(output_dtype)
        inverse_transfrom = transforms.ConvertImageDtype(input_dtype)
        output_image = inverse_transfrom(transform(input_image))

        actual_min, actual_max = output_image.tolist()
        desired_min, desired_max = 0, input_max

        assert actual_min == desired_min
        assert actual_max == desired_max


@pytest.mark.skipif(accimage is None, reason="accimage not available")
class TestAccImage:
    def test_accimage_to_tensor(self):
        trans = transforms.PILToTensor()

        expected_output = trans(Image.open(GRACE_HOPPER).convert("RGB"))
        output = trans(accimage.Image(GRACE_HOPPER))

        torch.testing.assert_close(output, expected_output)

    def test_accimage_pil_to_tensor(self):
        trans = transforms.PILToTensor()

        expected_output = trans(Image.open(GRACE_HOPPER).convert("RGB"))
        output = trans(accimage.Image(GRACE_HOPPER))

        assert expected_output.size() == output.size()
        torch.testing.assert_close(output, expected_output)

    def test_accimage_resize(self):
        trans = transforms.Compose(
            [
                transforms.Resize(256, interpolation=Image.LINEAR),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float),
            ]
        )

        # Checking if Compose, Resize and ToTensor can be printed as string
        trans.__repr__()

        expected_output = trans(Image.open(GRACE_HOPPER).convert("RGB"))
        output = trans(accimage.Image(GRACE_HOPPER))

        assert expected_output.size() == output.size()
        assert np.abs((expected_output - output).mean()) < 1e-3
        assert (expected_output - output).var() < 1e-5
        # note the high absolute tolerance
        torch.testing.assert_close(output.numpy(), expected_output.numpy(), rtol=1e-5, atol=5e-2)

    def test_accimage_crop(self):
        trans = transforms.Compose(
            [transforms.CenterCrop(256), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float)]
        )

        # Checking if Compose, CenterCrop and ToTensor can be printed as string
        trans.__repr__()

        expected_output = trans(Image.open(GRACE_HOPPER).convert("RGB"))
        output = trans(accimage.Image(GRACE_HOPPER))

        assert expected_output.size() == output.size()
        torch.testing.assert_close(output, expected_output)


class TestToTensor:
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_to_tensor(self, channels):
        height, width = 4, 4
        trans = transforms.ToTensor()
        np_rng = np.random.RandomState(0)

        input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
        img = transforms.ToPILImage()(input_data)
        output = trans(img)
        torch.testing.assert_close(output, input_data)

        ndarray = np_rng.randint(low=0, high=255, size=(height, width, channels)).astype(np.uint8)
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        torch.testing.assert_close(output.numpy(), expected_output, check_dtype=False)

        ndarray = np_rng.rand(height, width, channels).astype(np.float32)
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1))
        torch.testing.assert_close(output.numpy(), expected_output, check_dtype=False)

        # separate test for mode '1' PIL images
        input_data = torch.ByteTensor(1, height, width).bernoulli_()
        img = transforms.ToPILImage()(input_data.mul(255)).convert("1")
        output = trans(img)
        torch.testing.assert_close(input_data, output, check_dtype=False)

    def test_to_tensor_errors(self):
        height, width = 4, 4
        trans = transforms.ToTensor()
        np_rng = np.random.RandomState(0)

        with pytest.raises(TypeError):
            trans(np_rng.rand(1, height, width).tolist())

        with pytest.raises(ValueError):
            trans(np_rng.rand(height))

        with pytest.raises(ValueError):
            trans(np_rng.rand(1, 1, height, width))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float, torch.double])
    def test_to_tensor_with_other_default_dtypes(self, dtype):
        np_rng = np.random.RandomState(0)
        current_def_dtype = torch.get_default_dtype()

        t = transforms.ToTensor()
        np_arr = np_rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(np_arr)

        torch.set_default_dtype(dtype)
        res = t(img)
        assert res.dtype == dtype, f"{res.dtype} vs {dtype}"

        torch.set_default_dtype(current_def_dtype)

    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_pil_to_tensor(self, channels):
        height, width = 4, 4
        trans = transforms.PILToTensor()
        np_rng = np.random.RandomState(0)

        input_data = torch.ByteTensor(channels, height, width).random_(0, 255)
        img = transforms.ToPILImage()(input_data)
        output = trans(img)
        torch.testing.assert_close(input_data, output)

        input_data = np_rng.randint(low=0, high=255, size=(height, width, channels)).astype(np.uint8)
        img = transforms.ToPILImage()(input_data)
        output = trans(img)
        expected_output = input_data.transpose((2, 0, 1))
        torch.testing.assert_close(output.numpy(), expected_output)

        input_data = torch.as_tensor(np_rng.rand(channels, height, width).astype(np.float32))
        img = transforms.ToPILImage()(input_data)  # CHW -> HWC and (* 255).byte()
        output = trans(img)  # HWC -> CHW
        expected_output = (input_data * 255).byte()
        torch.testing.assert_close(output, expected_output)

        # separate test for mode '1' PIL images
        input_data = torch.ByteTensor(1, height, width).bernoulli_()
        img = transforms.ToPILImage()(input_data.mul(255)).convert("1")
        output = trans(img).view(torch.uint8).bool().to(torch.uint8)
        torch.testing.assert_close(input_data, output)

    def test_pil_to_tensor_errors(self):
        height, width = 4, 4
        trans = transforms.PILToTensor()
        np_rng = np.random.RandomState(0)

        with pytest.raises(TypeError):
            trans(np_rng.rand(1, height, width).tolist())

        with pytest.raises(TypeError):
            trans(np_rng.rand(1, height, width))


def test_randomresized_params():
    height = random.randint(24, 32) * 2
    width = random.randint(24, 32) * 2
    img = torch.ones(3, height, width)
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(img)
    size = 100
    epsilon = 0.05
    min_scale = 0.25
    for _ in range(10):
        scale_min = max(round(random.random(), 2), min_scale)
        scale_range = (scale_min, scale_min + round(random.random(), 2))
        aspect_min = max(round(random.random(), 2), epsilon)
        aspect_ratio_range = (aspect_min, aspect_min + round(random.random(), 2))
        randresizecrop = transforms.RandomResizedCrop(size, scale_range, aspect_ratio_range, antialias=True)
        i, j, h, w = randresizecrop.get_params(img, scale_range, aspect_ratio_range)
        aspect_ratio_obtained = w / h
        assert (
            min(aspect_ratio_range) - epsilon <= aspect_ratio_obtained
            and aspect_ratio_obtained <= max(aspect_ratio_range) + epsilon
        ) or aspect_ratio_obtained == 1.0
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(h, int)
        assert isinstance(w, int)


@pytest.mark.parametrize(
    "height, width",
    [
        # height, width
        # square image
        (28, 28),
        (27, 27),
        # rectangular image: h < w
        (28, 34),
        (29, 35),
        # rectangular image: h > w
        (34, 28),
        (35, 29),
    ],
)
@pytest.mark.parametrize(
    "osize",
    [
        # single integer
        22,
        27,
        28,
        36,
        # single integer in tuple/list
        [
            22,
        ],
        (27,),
    ],
)
@pytest.mark.parametrize("max_size", (None, 37, 1000))
def test_resize(height, width, osize, max_size):
    img = Image.new("RGB", size=(width, height), color=127)

    t = transforms.Resize(osize, max_size=max_size, antialias=True)
    result = t(img)

    msg = f"{height}, {width} - {osize} - {max_size}"
    osize = osize[0] if isinstance(osize, (list, tuple)) else osize
    # If size is an int, smaller edge of the image will be matched to this number.
    # i.e, if height > width, then image will be rescaled to (size * height / width, size).
    if height < width:
        exp_w, exp_h = (int(osize * width / height), osize)  # (w, h)
        if max_size is not None and max_size < exp_w:
            exp_w, exp_h = max_size, int(max_size * exp_h / exp_w)
        assert result.size == (exp_w, exp_h), msg
    elif width < height:
        exp_w, exp_h = (osize, int(osize * height / width))  # (w, h)
        if max_size is not None and max_size < exp_h:
            exp_w, exp_h = int(max_size * exp_w / exp_h), max_size
        assert result.size == (exp_w, exp_h), msg
    else:
        exp_w, exp_h = (osize, osize)  # (w, h)
        if max_size is not None and max_size < osize:
            exp_w, exp_h = max_size, max_size
        assert result.size == (exp_w, exp_h), msg


@pytest.mark.parametrize(
    "height, width",
    [
        # height, width
        # square image
        (28, 28),
        (27, 27),
        # rectangular image: h < w
        (28, 34),
        (29, 35),
        # rectangular image: h > w
        (34, 28),
        (35, 29),
    ],
)
@pytest.mark.parametrize(
    "osize",
    [
        # two integers sequence output
        [22, 22],
        [22, 28],
        [22, 36],
        [27, 22],
        [36, 22],
        [28, 28],
        [28, 37],
        [37, 27],
        [37, 37],
    ],
)
def test_resize_sequence_output(height, width, osize):
    img = Image.new("RGB", size=(width, height), color=127)
    oheight, owidth = osize

    t = transforms.Resize(osize, antialias=True)
    result = t(img)

    assert (owidth, oheight) == result.size


def test_resize_antialias_error():
    osize = [37, 37]
    img = Image.new("RGB", size=(35, 29), color=127)

    with pytest.warns(UserWarning, match=r"Anti-alias option is always applied for PIL Image input"):
        t = transforms.Resize(osize, antialias=False)
        t(img)


@pytest.mark.parametrize("height, width", ((32, 64), (64, 32)))
def test_resize_size_equals_small_edge_size(height, width):
    # Non-regression test for https://github.com/pytorch/vision/issues/5405
    # max_size used to be ignored if size == small_edge_size
    max_size = 40
    img = Image.new("RGB", size=(width, height), color=127)

    small_edge = min(height, width)
    t = transforms.Resize(small_edge, max_size=max_size, antialias=True)
    result = t(img)
    assert max(result.size) == max_size


def test_resize_equal_input_output_sizes():
    # Regression test for https://github.com/pytorch/vision/issues/7518
    height, width = 28, 27
    img = Image.new("RGB", size=(width, height))

    t = transforms.Resize((height, width), antialias=True)
    result = t(img)
    assert result is img


class TestPad:
    @pytest.mark.parametrize("fill", [85, 85.0])
    def test_pad(self, fill):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = torch.ones(3, height, width, dtype=torch.uint8)
        padding = random.randint(1, 20)
        result = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Pad(padding, fill=fill),
                transforms.PILToTensor(),
            ]
        )(img)
        assert result.size(1) == height + 2 * padding
        assert result.size(2) == width + 2 * padding
        # check that all elements in the padded region correspond
        # to the pad value
        h_padded = result[:, :padding, :]
        w_padded = result[:, :, :padding]
        torch.testing.assert_close(h_padded, torch.full_like(h_padded, fill_value=fill), rtol=0.0, atol=0.0)
        torch.testing.assert_close(w_padded, torch.full_like(w_padded, fill_value=fill), rtol=0.0, atol=0.0)
        pytest.raises(ValueError, transforms.Pad(padding, fill=(1, 2)), transforms.ToPILImage()(img))

    def test_pad_with_tuple_of_pad_values(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = transforms.ToPILImage()(torch.ones(3, height, width))

        padding = tuple(random.randint(1, 20) for _ in range(2))
        output = transforms.Pad(padding)(img)
        assert output.size == (width + padding[0] * 2, height + padding[1] * 2)

        padding = [random.randint(1, 20) for _ in range(4)]
        output = transforms.Pad(padding)(img)
        assert output.size[0] == width + padding[0] + padding[2]
        assert output.size[1] == height + padding[1] + padding[3]

        # Checking if Padding can be printed as string
        transforms.Pad(padding).__repr__()

    def test_pad_with_non_constant_padding_modes(self):
        """Unit tests for edge, reflect, symmetric padding"""
        img = torch.zeros(3, 27, 27).byte()
        img[:, :, 0] = 1  # Constant value added to leftmost edge
        img = transforms.ToPILImage()(img)
        img = F.pad(img, 1, (200, 200, 200))

        # pad 3 to all sidess
        edge_padded_img = F.pad(img, 3, padding_mode="edge")
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # edge_pad, edge_pad, edge_pad, constant_pad, constant value added to leftmost edge, 0
        edge_middle_slice = np.asarray(edge_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert_equal(edge_middle_slice, np.asarray([200, 200, 200, 200, 1, 0], dtype=np.uint8))
        assert transforms.PILToTensor()(edge_padded_img).size() == (3, 35, 35)

        # Pad 3 to left/right, 2 to top/bottom
        reflect_padded_img = F.pad(img, (3, 2), padding_mode="reflect")
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # reflect_pad, reflect_pad, reflect_pad, constant_pad, constant value added to leftmost edge, 0
        reflect_middle_slice = np.asarray(reflect_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert_equal(reflect_middle_slice, np.asarray([0, 0, 1, 200, 1, 0], dtype=np.uint8))
        assert transforms.PILToTensor()(reflect_padded_img).size() == (3, 33, 35)

        # Pad 3 to left, 2 to top, 2 to right, 1 to bottom
        symmetric_padded_img = F.pad(img, (3, 2, 2, 1), padding_mode="symmetric")
        # First 6 elements of leftmost edge in the middle of the image, values are in order:
        # sym_pad, sym_pad, sym_pad, constant_pad, constant value added to leftmost edge, 0
        symmetric_middle_slice = np.asarray(symmetric_padded_img).transpose(2, 0, 1)[0][17][:6]
        assert_equal(symmetric_middle_slice, np.asarray([0, 1, 200, 200, 1, 0], dtype=np.uint8))
        assert transforms.PILToTensor()(symmetric_padded_img).size() == (3, 32, 34)

        # Check negative padding explicitly for symmetric case, since it is not
        # implemented for tensor case to compare to
        # Crop 1 to left, pad 2 to top, pad 3 to right, crop 3 to bottom
        symmetric_padded_img_neg = F.pad(img, (-1, 2, 3, -3), padding_mode="symmetric")
        symmetric_neg_middle_left = np.asarray(symmetric_padded_img_neg).transpose(2, 0, 1)[0][17][:3]
        symmetric_neg_middle_right = np.asarray(symmetric_padded_img_neg).transpose(2, 0, 1)[0][17][-4:]
        assert_equal(symmetric_neg_middle_left, np.asarray([1, 0, 0], dtype=np.uint8))
        assert_equal(symmetric_neg_middle_right, np.asarray([200, 200, 0, 0], dtype=np.uint8))
        assert transforms.PILToTensor()(symmetric_padded_img_neg).size() == (3, 28, 31)

    def test_pad_raises_with_invalid_pad_sequence_len(self):
        with pytest.raises(ValueError):
            transforms.Pad(())

        with pytest.raises(ValueError):
            transforms.Pad((1, 2, 3))

        with pytest.raises(ValueError):
            transforms.Pad((1, 2, 3, 4, 5))

    def test_pad_with_mode_F_images(self):
        pad = 2
        transform = transforms.Pad(pad)

        img = Image.new("F", (10, 10))
        padded_img = transform(img)
        assert_equal(padded_img.size, [edge_size + 2 * pad for edge_size in img.size])


@pytest.mark.parametrize(
    "fn, trans, kwargs",
    [
        (F.invert, transforms.RandomInvert, {}),
        (F.posterize, transforms.RandomPosterize, {"bits": 4}),
        (F.solarize, transforms.RandomSolarize, {"threshold": 192}),
        (F.adjust_sharpness, transforms.RandomAdjustSharpness, {"sharpness_factor": 2.0}),
        (F.autocontrast, transforms.RandomAutocontrast, {}),
        (F.equalize, transforms.RandomEqualize, {}),
        (F.vflip, transforms.RandomVerticalFlip, {}),
        (F.hflip, transforms.RandomHorizontalFlip, {}),
        (partial(F.to_grayscale, num_output_channels=3), transforms.RandomGrayscale, {}),
    ],
)
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("p", (0, 1))
def test_randomness(fn, trans, kwargs, seed, p):
    torch.manual_seed(seed)
    img = transforms.ToPILImage()(torch.rand(3, 16, 18))

    expected_transformed_img = fn(img, **kwargs)
    randomly_transformed_img = trans(p=p, **kwargs)(img)

    if p == 0:
        assert randomly_transformed_img == img
    elif p == 1:
        assert randomly_transformed_img == expected_transformed_img

    trans(**kwargs).__repr__()


def test_autocontrast_equal_minmax():
    img_tensor = torch.tensor([[[10]], [[128]], [[245]]], dtype=torch.uint8).expand(3, 32, 32)
    img_pil = F.to_pil_image(img_tensor)

    img_tensor = F.autocontrast(img_tensor)
    img_pil = F.autocontrast(img_pil)
    torch.testing.assert_close(img_tensor, F.pil_to_tensor(img_pil))


class TestToPil:
    def _get_1_channel_tensor_various_types():
        img_data_float = torch.Tensor(1, 4, 4).uniform_()
        expected_output = img_data_float.mul(255).int().float().div(255).numpy()
        yield img_data_float, expected_output, "L"

        img_data_byte = torch.ByteTensor(1, 4, 4).random_(0, 255)
        expected_output = img_data_byte.float().div(255.0).numpy()
        yield img_data_byte, expected_output, "L"

        img_data_short = torch.ShortTensor(1, 4, 4).random_()
        expected_output = img_data_short.numpy()
        yield img_data_short, expected_output, "I;16" if sys.byteorder == "little" else "I;16B"

        img_data_int = torch.IntTensor(1, 4, 4).random_()
        expected_output = img_data_int.numpy()
        yield img_data_int, expected_output, "I"

    def _get_2d_tensor_various_types():
        img_data_float = torch.Tensor(4, 4).uniform_()
        expected_output = img_data_float.mul(255).int().float().div(255).numpy()
        yield img_data_float, expected_output, "L"

        img_data_byte = torch.ByteTensor(4, 4).random_(0, 255)
        expected_output = img_data_byte.float().div(255.0).numpy()
        yield img_data_byte, expected_output, "L"

        img_data_short = torch.ShortTensor(4, 4).random_()
        expected_output = img_data_short.numpy()
        yield img_data_short, expected_output, "I;16" if sys.byteorder == "little" else "I;16B"

        img_data_int = torch.IntTensor(4, 4).random_()
        expected_output = img_data_int.numpy()
        yield img_data_int, expected_output, "I"

    @pytest.mark.parametrize("with_mode", [False, True])
    @pytest.mark.parametrize("img_data, expected_output, expected_mode", _get_1_channel_tensor_various_types())
    def test_1_channel_tensor_to_pil_image(self, with_mode, img_data, expected_output, expected_mode):
        transform = transforms.ToPILImage(mode=expected_mode) if with_mode else transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img = transform(img_data)
        assert img.mode == expected_mode
        torch.testing.assert_close(expected_output, to_tensor(img).numpy())

    def test_1_channel_float_tensor_to_pil_image(self):
        img_data = torch.Tensor(1, 4, 4).uniform_()
        # 'F' mode for torch.FloatTensor
        img_F_mode = transforms.ToPILImage(mode="F")(img_data)
        assert img_F_mode.mode == "F"
        torch.testing.assert_close(
            np.array(Image.fromarray(img_data.squeeze(0).numpy(), mode="F")), np.array(img_F_mode)
        )

    @pytest.mark.parametrize("with_mode", [False, True])
    @pytest.mark.parametrize(
        "img_data, expected_mode",
        [
            (torch.Tensor(4, 4, 1).uniform_().numpy(), "L"),
            (torch.ByteTensor(4, 4, 1).random_(0, 255).numpy(), "L"),
            (torch.ShortTensor(4, 4, 1).random_().numpy(), "I;16" if sys.byteorder == "little" else "I;16B"),
            (torch.IntTensor(4, 4, 1).random_().numpy(), "I"),
        ],
    )
    def test_1_channel_ndarray_to_pil_image(self, with_mode, img_data, expected_mode):
        transform = transforms.ToPILImage(mode=expected_mode) if with_mode else transforms.ToPILImage()
        img = transform(img_data)
        assert img.mode == expected_mode
        if np.issubdtype(img_data.dtype, np.floating):
            img_data = (img_data * 255).astype(np.uint8)
        # note: we explicitly convert img's dtype because pytorch doesn't support uint16
        # and otherwise assert_close wouldn't be able to construct a tensor from the uint16 array
        torch.testing.assert_close(img_data[:, :, 0], np.asarray(img).astype(img_data.dtype))

    @pytest.mark.parametrize("expected_mode", [None, "LA"])
    def test_2_channel_ndarray_to_pil_image(self, expected_mode):
        img_data = torch.ByteTensor(4, 4, 2).random_(0, 255).numpy()

        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "LA"  # default should assume LA
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode
        split = img.split()
        for i in range(2):
            torch.testing.assert_close(img_data[:, :, i], np.asarray(split[i]))

    def test_2_channel_ndarray_to_pil_image_error(self):
        img_data = torch.ByteTensor(4, 4, 2).random_(0, 255).numpy()
        transforms.ToPILImage().__repr__()

        # should raise if we try a mode for 4 or 1 or 3 channel images
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="RGBA")(img_data)
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="RGB")(img_data)

    @pytest.mark.parametrize("expected_mode", [None, "LA"])
    def test_2_channel_tensor_to_pil_image(self, expected_mode):
        img_data = torch.Tensor(2, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)
        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "LA"  # default should assume LA
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode

        split = img.split()
        for i in range(2):
            torch.testing.assert_close(expected_output[i].numpy(), F.to_tensor(split[i]).squeeze(0).numpy())

    def test_2_channel_tensor_to_pil_image_error(self):
        img_data = torch.Tensor(2, 4, 4).uniform_()

        # should raise if we try a mode for 4 or 1 or 3 channel images
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="RGBA")(img_data)
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=r"Only modes \['LA'\] are supported for 2D inputs"):
            transforms.ToPILImage(mode="RGB")(img_data)

    @pytest.mark.parametrize("with_mode", [False, True])
    @pytest.mark.parametrize("img_data, expected_output, expected_mode", _get_2d_tensor_various_types())
    def test_2d_tensor_to_pil_image(self, with_mode, img_data, expected_output, expected_mode):
        transform = transforms.ToPILImage(mode=expected_mode) if with_mode else transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img = transform(img_data)
        assert img.mode == expected_mode
        torch.testing.assert_close(expected_output, to_tensor(img).numpy()[0])

    @pytest.mark.parametrize("with_mode", [False, True])
    @pytest.mark.parametrize(
        "img_data, expected_mode",
        [
            (torch.Tensor(4, 4).uniform_().numpy(), "L"),
            (torch.ByteTensor(4, 4).random_(0, 255).numpy(), "L"),
            (torch.ShortTensor(4, 4).random_().numpy(), "I;16" if sys.byteorder == "little" else "I;16B"),
            (torch.IntTensor(4, 4).random_().numpy(), "I"),
        ],
    )
    def test_2d_ndarray_to_pil_image(self, with_mode, img_data, expected_mode):
        transform = transforms.ToPILImage(mode=expected_mode) if with_mode else transforms.ToPILImage()
        img = transform(img_data)
        assert img.mode == expected_mode
        if np.issubdtype(img_data.dtype, np.floating):
            img_data = (img_data * 255).astype(np.uint8)
        np.testing.assert_allclose(img_data, img)

    @pytest.mark.parametrize("expected_mode", [None, "RGB", "HSV", "YCbCr"])
    def test_3_channel_tensor_to_pil_image(self, expected_mode):
        img_data = torch.Tensor(3, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)

        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "RGB"  # default should assume RGB
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode
        split = img.split()
        for i in range(3):
            torch.testing.assert_close(expected_output[i].numpy(), F.to_tensor(split[i]).squeeze(0).numpy())

    def test_3_channel_tensor_to_pil_image_error(self):
        img_data = torch.Tensor(3, 4, 4).uniform_()
        error_message_3d = r"Only modes \['RGB', 'YCbCr', 'HSV'\] are supported for 3D inputs"
        # should raise if we try a mode for 4 or 1 or 2 channel images
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="RGBA")(img_data)
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="LA")(img_data)

        with pytest.raises(ValueError, match=r"pic should be 2/3 dimensional. Got \d+ dimensions."):
            transforms.ToPILImage()(torch.Tensor(1, 3, 4, 4).uniform_())

    @pytest.mark.parametrize("expected_mode", [None, "RGB", "HSV", "YCbCr"])
    def test_3_channel_ndarray_to_pil_image(self, expected_mode):
        img_data = torch.ByteTensor(4, 4, 3).random_(0, 255).numpy()

        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "RGB"  # default should assume RGB
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode
        split = img.split()
        for i in range(3):
            torch.testing.assert_close(img_data[:, :, i], np.asarray(split[i]))

    def test_3_channel_ndarray_to_pil_image_error(self):
        img_data = torch.ByteTensor(4, 4, 3).random_(0, 255).numpy()

        # Checking if ToPILImage can be printed as string
        transforms.ToPILImage().__repr__()

        error_message_3d = r"Only modes \['RGB', 'YCbCr', 'HSV'\] are supported for 3D inputs"
        # should raise if we try a mode for 4 or 1 or 2 channel images
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="RGBA")(img_data)
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=error_message_3d):
            transforms.ToPILImage(mode="LA")(img_data)

    @pytest.mark.parametrize("expected_mode", [None, "RGBA", "CMYK", "RGBX"])
    def test_4_channel_tensor_to_pil_image(self, expected_mode):
        img_data = torch.Tensor(4, 4, 4).uniform_()
        expected_output = img_data.mul(255).int().float().div(255)

        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "RGBA"  # default should assume RGBA
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode

        split = img.split()
        for i in range(4):
            torch.testing.assert_close(expected_output[i].numpy(), F.to_tensor(split[i]).squeeze(0).numpy())

    def test_4_channel_tensor_to_pil_image_error(self):
        img_data = torch.Tensor(4, 4, 4).uniform_()

        error_message_4d = r"Only modes \['RGBA', 'CMYK', 'RGBX'\] are supported for 4D inputs"
        # should raise if we try a mode for 3 or 1 or 2 channel images
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="RGB")(img_data)
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="LA")(img_data)

    @pytest.mark.parametrize("expected_mode", [None, "RGBA", "CMYK", "RGBX"])
    def test_4_channel_ndarray_to_pil_image(self, expected_mode):
        img_data = torch.ByteTensor(4, 4, 4).random_(0, 255).numpy()

        if expected_mode is None:
            img = transforms.ToPILImage()(img_data)
            assert img.mode == "RGBA"  # default should assume RGBA
        else:
            img = transforms.ToPILImage(mode=expected_mode)(img_data)
            assert img.mode == expected_mode
        split = img.split()
        for i in range(4):
            torch.testing.assert_close(img_data[:, :, i], np.asarray(split[i]))

    def test_4_channel_ndarray_to_pil_image_error(self):
        img_data = torch.ByteTensor(4, 4, 4).random_(0, 255).numpy()

        error_message_4d = r"Only modes \['RGBA', 'CMYK', 'RGBX'\] are supported for 4D inputs"
        # should raise if we try a mode for 3 or 1 or 2 channel images
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="RGB")(img_data)
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="P")(img_data)
        with pytest.raises(ValueError, match=error_message_4d):
            transforms.ToPILImage(mode="LA")(img_data)

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transforms.ToPILImage()
        reg_msg = r"Input type \w+ is not supported"
        with pytest.raises(TypeError, match=reg_msg):
            trans(np.ones([4, 4, 1], np.int64))
        with pytest.raises(TypeError, match=reg_msg):
            trans(np.ones([4, 4, 1], np.uint16))
        with pytest.raises(TypeError, match=reg_msg):
            trans(np.ones([4, 4, 1], np.uint32))

        with pytest.raises(ValueError, match=r"pic should be 2/3 dimensional. Got \d+ dimensions."):
            transforms.ToPILImage()(np.ones([1, 4, 4, 3]))
        with pytest.raises(ValueError, match=r"pic should not have > 4 channels. Got \d+ channels."):
            transforms.ToPILImage()(np.ones([4, 4, 6]))

    def test_tensor_bad_types_to_pil_image(self):
        with pytest.raises(ValueError, match=r"pic should be 2/3 dimensional. Got \d+ dimensions."):
            transforms.ToPILImage()(torch.ones(1, 3, 4, 4))
        with pytest.raises(ValueError, match=r"pic should not have > 4 channels. Got \d+ channels."):
            transforms.ToPILImage()(torch.ones(6, 4, 4))


def test_adjust_brightness():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_brightness(x_pil, 1)
    y_np = np.array(y_pil)
    torch.testing.assert_close(y_np, x_np)

    # test 1
    y_pil = F.adjust_brightness(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 2
    y_pil = F.adjust_brightness(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)


def test_adjust_contrast():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_contrast(x_pil, 1)
    y_np = np.array(y_pil)
    torch.testing.assert_close(y_np, x_np)

    # test 1
    y_pil = F.adjust_contrast(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [43, 45, 49, 70, 110, 156, 61, 47, 160, 88, 170, 43]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 2
    y_pil = F.adjust_contrast(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 0, 0, 22, 184, 255, 0, 0, 255, 94, 255, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)


def test_adjust_hue():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    with pytest.raises(ValueError):
        F.adjust_hue(x_pil, -0.7)
        F.adjust_hue(x_pil, 1)

    # test 0: almost same as x_data but not exact.
    # probably because hsv <-> rgb floating point ops
    y_pil = F.adjust_hue(x_pil, 0)
    y_np = np.array(y_pil)
    y_ans = [0, 5, 13, 54, 139, 226, 35, 8, 234, 91, 255, 1]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 1
    y_pil = F.adjust_hue(x_pil, 0.25)
    y_np = np.array(y_pil)
    y_ans = [13, 0, 12, 224, 54, 226, 234, 8, 99, 1, 222, 255]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 2
    y_pil = F.adjust_hue(x_pil, -0.25)
    y_np = np.array(y_pil)
    y_ans = [0, 13, 2, 54, 226, 58, 8, 234, 152, 255, 43, 1]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)


def test_adjust_sharpness():
    x_shape = [4, 4, 3]
    x_data = [
        75,
        121,
        114,
        105,
        97,
        107,
        105,
        32,
        66,
        111,
        117,
        114,
        99,
        104,
        97,
        0,
        0,
        65,
        108,
        101,
        120,
        97,
        110,
        100,
        101,
        114,
        32,
        86,
        114,
        121,
        110,
        105,
        111,
        116,
        105,
        115,
        0,
        0,
        73,
        32,
        108,
        111,
        118,
        101,
        32,
        121,
        111,
        117,
    ]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_sharpness(x_pil, 1)
    y_np = np.array(y_pil)
    torch.testing.assert_close(y_np, x_np)

    # test 1
    y_pil = F.adjust_sharpness(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [
        75,
        121,
        114,
        105,
        97,
        107,
        105,
        32,
        66,
        111,
        117,
        114,
        99,
        104,
        97,
        30,
        30,
        74,
        103,
        96,
        114,
        97,
        110,
        100,
        101,
        114,
        32,
        81,
        103,
        108,
        102,
        101,
        107,
        116,
        105,
        115,
        0,
        0,
        73,
        32,
        108,
        111,
        118,
        101,
        32,
        121,
        111,
        117,
    ]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 2
    y_pil = F.adjust_sharpness(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [
        75,
        121,
        114,
        105,
        97,
        107,
        105,
        32,
        66,
        111,
        117,
        114,
        99,
        104,
        97,
        0,
        0,
        46,
        118,
        111,
        132,
        97,
        110,
        100,
        101,
        114,
        32,
        95,
        135,
        146,
        126,
        112,
        119,
        116,
        105,
        115,
        0,
        0,
        73,
        32,
        108,
        111,
        118,
        101,
        32,
        121,
        111,
        117,
    ]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 3
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")
    x_th = torch.tensor(x_np.transpose(2, 0, 1))
    y_pil = F.adjust_sharpness(x_pil, 2)
    y_np = np.array(y_pil).transpose(2, 0, 1)
    y_th = F.adjust_sharpness(x_th, 2)
    torch.testing.assert_close(y_np, y_th.numpy())


def test_adjust_gamma():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")

    # test 0
    y_pil = F.adjust_gamma(x_pil, 1)
    y_np = np.array(y_pil)
    torch.testing.assert_close(y_np, x_np)

    # test 1
    y_pil = F.adjust_gamma(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [0, 35, 57, 117, 186, 241, 97, 45, 245, 152, 255, 16]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)

    # test 2
    y_pil = F.adjust_gamma(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 0, 0, 11, 71, 201, 5, 0, 215, 31, 255, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    torch.testing.assert_close(y_np, y_ans)


def test_adjusts_L_mode():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_rgb = Image.fromarray(x_np, mode="RGB")

    x_l = x_rgb.convert("L")
    assert F.adjust_brightness(x_l, 2).mode == "L"
    assert F.adjust_saturation(x_l, 2).mode == "L"
    assert F.adjust_contrast(x_l, 2).mode == "L"
    assert F.adjust_hue(x_l, 0.4).mode == "L"
    assert F.adjust_sharpness(x_l, 2).mode == "L"
    assert F.adjust_gamma(x_l, 0.5).mode == "L"


def test_rotate():
    x = np.zeros((100, 100, 3), dtype=np.uint8)
    x[40, 40] = [255, 255, 255]

    with pytest.raises(TypeError, match=r"img should be PIL Image"):
        F.rotate(x, 10)

    img = F.to_pil_image(x)

    result = F.rotate(img, 45)
    assert result.size == (100, 100)
    r, c, ch = np.where(result)
    assert all(x in r for x in [49, 50])
    assert all(x in c for x in [36])
    assert all(x in ch for x in [0, 1, 2])

    result = F.rotate(img, 45, expand=True)
    assert result.size == (142, 142)
    r, c, ch = np.where(result)
    assert all(x in r for x in [70, 71])
    assert all(x in c for x in [57])
    assert all(x in ch for x in [0, 1, 2])

    result = F.rotate(img, 45, center=(40, 40))
    assert result.size == (100, 100)
    r, c, ch = np.where(result)
    assert all(x in r for x in [40])
    assert all(x in c for x in [40])
    assert all(x in ch for x in [0, 1, 2])

    result_a = F.rotate(img, 90)
    result_b = F.rotate(img, -270)

    assert_equal(np.array(result_a), np.array(result_b))


@pytest.mark.parametrize("mode", ["L", "RGB", "F"])
def test_rotate_fill(mode):
    img = F.to_pil_image(np.ones((100, 100, 3), dtype=np.uint8) * 255, "RGB")

    num_bands = len(mode)
    wrong_num_bands = num_bands + 1
    fill = 127

    img_conv = img.convert(mode)
    img_rot = F.rotate(img_conv, 45.0, fill=fill)
    pixel = img_rot.getpixel((0, 0))

    if not isinstance(pixel, tuple):
        pixel = (pixel,)
    assert pixel == tuple([fill] * num_bands)

    with pytest.raises(ValueError):
        F.rotate(img_conv, 45.0, fill=tuple([fill] * wrong_num_bands))


def test_gaussian_blur_asserts():
    np_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img = F.to_pil_image(np_img, "RGB")

    with pytest.raises(ValueError, match=r"If kernel_size is a sequence its length should be 2"):
        F.gaussian_blur(img, [3])
    with pytest.raises(ValueError, match=r"If kernel_size is a sequence its length should be 2"):
        F.gaussian_blur(img, [3, 3, 3])
    with pytest.raises(ValueError, match=r"Kernel size should be a tuple/list of two integers"):
        transforms.GaussianBlur([3, 3, 3])

    with pytest.raises(ValueError, match=r"kernel_size should have odd and positive integers"):
        F.gaussian_blur(img, [4, 4])
    with pytest.raises(ValueError, match=r"Kernel size value should be an odd and positive number"):
        transforms.GaussianBlur([4, 4])

    with pytest.raises(ValueError, match=r"kernel_size should have odd and positive integers"):
        F.gaussian_blur(img, [-3, -3])
    with pytest.raises(ValueError, match=r"Kernel size value should be an odd and positive number"):
        transforms.GaussianBlur([-3, -3])

    with pytest.raises(ValueError, match=r"If sigma is a sequence, its length should be 2"):
        F.gaussian_blur(img, 3, [1, 1, 1])
    with pytest.raises(ValueError, match=r"sigma should be a single number or a list/tuple with length 2"):
        transforms.GaussianBlur(3, [1, 1, 1])

    with pytest.raises(ValueError, match=r"sigma should have positive values"):
        F.gaussian_blur(img, 3, -1.0)
    with pytest.raises(ValueError, match=r"If sigma is a single number, it must be positive"):
        transforms.GaussianBlur(3, -1.0)

    with pytest.raises(TypeError, match=r"kernel_size should be int or a sequence of integers"):
        F.gaussian_blur(img, "kernel_size_string")
    with pytest.raises(ValueError, match=r"Kernel size should be a tuple/list of two integers"):
        transforms.GaussianBlur("kernel_size_string")

    with pytest.raises(TypeError, match=r"sigma should be either float or sequence of floats"):
        F.gaussian_blur(img, 3, "sigma_string")
    with pytest.raises(ValueError, match=r"sigma should be a single number or a list/tuple with length 2"):
        transforms.GaussianBlur(3, "sigma_string")


def test_lambda():
    trans = transforms.Lambda(lambda x: x.add(10))
    x = torch.randn(10)
    y = trans(x)
    assert_equal(y, torch.add(x, 10))

    trans = transforms.Lambda(lambda x: x.add_(10))
    x = torch.randn(10)
    y = trans(x)
    assert_equal(y, x)

    # Checking if Lambda can be printed as string
    trans.__repr__()


def test_to_grayscale():
    """Unit tests for grayscale transform"""

    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")
    x_pil_2 = x_pil.convert("L")
    gray_np = np.array(x_pil_2)

    # Test Set: Grayscale an image with desired number of output channels
    # Case 1: RGB -> 1 channel grayscale
    trans1 = transforms.Grayscale(num_output_channels=1)
    gray_pil_1 = trans1(x_pil)
    gray_np_1 = np.array(gray_pil_1)
    assert gray_pil_1.mode == "L", "mode should be L"
    assert gray_np_1.shape == tuple(x_shape[0:2]), "should be 1 channel"
    assert_equal(gray_np, gray_np_1)

    # Case 2: RGB -> 3 channel grayscale
    trans2 = transforms.Grayscale(num_output_channels=3)
    gray_pil_2 = trans2(x_pil)
    gray_np_2 = np.array(gray_pil_2)
    assert gray_pil_2.mode == "RGB", "mode should be RGB"
    assert gray_np_2.shape == tuple(x_shape), "should be 3 channel"
    assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
    assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
    assert_equal(gray_np, gray_np_2[:, :, 0])

    # Case 3: 1 channel grayscale -> 1 channel grayscale
    trans3 = transforms.Grayscale(num_output_channels=1)
    gray_pil_3 = trans3(x_pil_2)
    gray_np_3 = np.array(gray_pil_3)
    assert gray_pil_3.mode == "L", "mode should be L"
    assert gray_np_3.shape == tuple(x_shape[0:2]), "should be 1 channel"
    assert_equal(gray_np, gray_np_3)

    # Case 4: 1 channel grayscale -> 3 channel grayscale
    trans4 = transforms.Grayscale(num_output_channels=3)
    gray_pil_4 = trans4(x_pil_2)
    gray_np_4 = np.array(gray_pil_4)
    assert gray_pil_4.mode == "RGB", "mode should be RGB"
    assert gray_np_4.shape == tuple(x_shape), "should be 3 channel"
    assert_equal(gray_np_4[:, :, 0], gray_np_4[:, :, 1])
    assert_equal(gray_np_4[:, :, 1], gray_np_4[:, :, 2])
    assert_equal(gray_np, gray_np_4[:, :, 0])

    # Checking if Grayscale can be printed as string
    trans4.__repr__()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("p", (0, 1))
def test_random_apply(p, seed):
    torch.manual_seed(seed)
    random_apply_transform = transforms.RandomApply([transforms.RandomRotation((45, 50))], p=p)
    img = transforms.ToPILImage()(torch.rand(3, 30, 40))
    out = random_apply_transform(img)
    if p == 0:
        assert out == img
    elif p == 1:
        assert out != img

    # Checking if RandomApply can be printed as string
    random_apply_transform.__repr__()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("proba_passthrough", (0, 1))
def test_random_choice(proba_passthrough, seed):
    random.seed(seed)  # RandomChoice relies on python builtin random.choice, not pytorch

    random_choice_transform = transforms.RandomChoice(
        [
            lambda x: x,  # passthrough
            transforms.RandomRotation((45, 50)),
        ],
        p=[proba_passthrough, 1 - proba_passthrough],
    )

    img = transforms.ToPILImage()(torch.rand(3, 30, 40))
    out = random_choice_transform(img)
    if proba_passthrough == 1:
        assert out == img
    elif proba_passthrough == 0:
        assert out != img

    # Checking if RandomChoice can be printed as string
    random_choice_transform.__repr__()


@pytest.mark.skipif(stats is None, reason="scipy.stats not available")
def test_random_order():
    random_state = random.getstate()
    random.seed(42)
    random_order_transform = transforms.RandomOrder([transforms.Resize(20, antialias=True), transforms.CenterCrop(10)])
    img = transforms.ToPILImage()(torch.rand(3, 25, 25))
    num_samples = 250
    num_normal_order = 0
    resize_crop_out = transforms.CenterCrop(10)(transforms.Resize(20, antialias=True)(img))
    for _ in range(num_samples):
        out = random_order_transform(img)
        if out == resize_crop_out:
            num_normal_order += 1

    p_value = stats.binomtest(num_normal_order, num_samples, p=0.5).pvalue
    random.setstate(random_state)
    assert p_value > 0.0001

    # Checking if RandomOrder can be printed as string
    random_order_transform.__repr__()


def test_linear_transformation():
    num_samples = 1000
    x = torch.randn(num_samples, 3, 10, 10)
    flat_x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    # compute principal components
    sigma = torch.mm(flat_x.t(), flat_x) / flat_x.size(0)
    u, s, _ = np.linalg.svd(sigma.numpy())
    zca_epsilon = 1e-10  # avoid division by 0
    d = torch.Tensor(np.diag(1.0 / np.sqrt(s + zca_epsilon)))
    u = torch.Tensor(u)
    principal_components = torch.mm(torch.mm(u, d), u.t())
    mean_vector = torch.sum(flat_x, dim=0) / flat_x.size(0)
    # initialize whitening matrix
    whitening = transforms.LinearTransformation(principal_components, mean_vector)
    # estimate covariance and mean using weak law of large number
    num_features = flat_x.size(1)
    cov = 0.0
    mean = 0.0
    for i in x:
        xwhite = whitening(i)
        xwhite = xwhite.view(1, -1).numpy()
        cov += np.dot(xwhite, xwhite.T) / num_features
        mean += np.sum(xwhite) / num_features
    # if rtol for std = 1e-3 then rtol for cov = 2e-3 as std**2 = cov
    torch.testing.assert_close(
        cov / num_samples, np.identity(1), rtol=2e-3, atol=1e-8, check_dtype=False, msg="cov not close to 1"
    )
    torch.testing.assert_close(
        mean / num_samples, 0, rtol=1e-3, atol=1e-8, check_dtype=False, msg="mean not close to 0"
    )

    # Checking if LinearTransformation can be printed as string
    whitening.__repr__()


@pytest.mark.parametrize("dtype", int_dtypes())
def test_max_value(dtype):

    assert F_t._max_value(dtype) == torch.iinfo(dtype).max
    # remove float testing as it can lead to errors such as
    # runtime error: 5.7896e+76 is outside the range of representable values of type 'float'
    # for dtype in float_dtypes():
    # self.assertGreater(F_t._max_value(dtype), torch.finfo(dtype).max)


@pytest.mark.xfail(
    reason="torch.iinfo() is not supported by torchscript. See https://github.com/pytorch/pytorch/issues/41492."
)
def test_max_value_iinfo():
    @torch.jit.script
    def max_value(image: torch.Tensor) -> int:
        return 1 if image.is_floating_point() else torch.iinfo(image.dtype).max


@pytest.mark.parametrize("should_vflip", [True, False])
@pytest.mark.parametrize("single_dim", [True, False])
def test_ten_crop(should_vflip, single_dim):
    to_pil_image = transforms.ToPILImage()
    h = random.randint(5, 25)
    w = random.randint(5, 25)
    crop_h = random.randint(1, h)
    crop_w = random.randint(1, w)
    if single_dim:
        crop_h = min(crop_h, crop_w)
        crop_w = crop_h
        transform = transforms.TenCrop(crop_h, vertical_flip=should_vflip)
        five_crop = transforms.FiveCrop(crop_h)
    else:
        transform = transforms.TenCrop((crop_h, crop_w), vertical_flip=should_vflip)
        five_crop = transforms.FiveCrop((crop_h, crop_w))

    img = to_pil_image(torch.FloatTensor(3, h, w).uniform_())
    results = transform(img)
    expected_output = five_crop(img)

    # Checking if FiveCrop and TenCrop can be printed as string
    transform.__repr__()
    five_crop.__repr__()

    if should_vflip:
        vflipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        expected_output += five_crop(vflipped_img)
    else:
        hflipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        expected_output += five_crop(hflipped_img)

    assert len(results) == 10
    assert results == expected_output


@pytest.mark.parametrize("single_dim", [True, False])
def test_five_crop(single_dim):
    to_pil_image = transforms.ToPILImage()
    h = random.randint(5, 25)
    w = random.randint(5, 25)
    crop_h = random.randint(1, h)
    crop_w = random.randint(1, w)
    if single_dim:
        crop_h = min(crop_h, crop_w)
        crop_w = crop_h
        transform = transforms.FiveCrop(crop_h)
    else:
        transform = transforms.FiveCrop((crop_h, crop_w))

    img = torch.FloatTensor(3, h, w).uniform_()

    results = transform(to_pil_image(img))

    assert len(results) == 5
    for crop in results:
        assert crop.size == (crop_w, crop_h)

    to_pil_image = transforms.ToPILImage()
    tl = to_pil_image(img[:, 0:crop_h, 0:crop_w])
    tr = to_pil_image(img[:, 0:crop_h, w - crop_w :])
    bl = to_pil_image(img[:, h - crop_h :, 0:crop_w])
    br = to_pil_image(img[:, h - crop_h :, w - crop_w :])
    center = transforms.CenterCrop((crop_h, crop_w))(to_pil_image(img))
    expected_output = (tl, tr, bl, br, center)
    assert results == expected_output


@pytest.mark.parametrize("policy", transforms.AutoAugmentPolicy)
@pytest.mark.parametrize("fill", [None, 85, (128, 128, 128)])
@pytest.mark.parametrize("grayscale", [True, False])
def test_autoaugment(policy, fill, grayscale):
    random.seed(42)
    img = Image.open(GRACE_HOPPER)
    if grayscale:
        img, fill = _get_grayscale_test_image(img, fill)
    transform = transforms.AutoAugment(policy=policy, fill=fill)
    for _ in range(100):
        img = transform(img)
    transform.__repr__()


@pytest.mark.parametrize("num_ops", [1, 2, 3])
@pytest.mark.parametrize("magnitude", [7, 9, 11])
@pytest.mark.parametrize("fill", [None, 85, (128, 128, 128)])
@pytest.mark.parametrize("grayscale", [True, False])
def test_randaugment(num_ops, magnitude, fill, grayscale):
    random.seed(42)
    img = Image.open(GRACE_HOPPER)
    if grayscale:
        img, fill = _get_grayscale_test_image(img, fill)
    transform = transforms.RandAugment(num_ops=num_ops, magnitude=magnitude, fill=fill)
    for _ in range(100):
        img = transform(img)
    transform.__repr__()


@pytest.mark.parametrize("fill", [None, 85, (128, 128, 128)])
@pytest.mark.parametrize("num_magnitude_bins", [10, 13, 30])
@pytest.mark.parametrize("grayscale", [True, False])
def test_trivialaugmentwide(fill, num_magnitude_bins, grayscale):
    random.seed(42)
    img = Image.open(GRACE_HOPPER)
    if grayscale:
        img, fill = _get_grayscale_test_image(img, fill)
    transform = transforms.TrivialAugmentWide(fill=fill, num_magnitude_bins=num_magnitude_bins)
    for _ in range(100):
        img = transform(img)
    transform.__repr__()


@pytest.mark.parametrize("fill", [None, 85, (128, 128, 128)])
@pytest.mark.parametrize("severity", [1, 10])
@pytest.mark.parametrize("mixture_width", [1, 2])
@pytest.mark.parametrize("chain_depth", [-1, 2])
@pytest.mark.parametrize("all_ops", [True, False])
@pytest.mark.parametrize("grayscale", [True, False])
def test_augmix(fill, severity, mixture_width, chain_depth, all_ops, grayscale):
    random.seed(42)
    img = Image.open(GRACE_HOPPER)
    if grayscale:
        img, fill = _get_grayscale_test_image(img, fill)
    transform = transforms.AugMix(
        fill=fill, severity=severity, mixture_width=mixture_width, chain_depth=chain_depth, all_ops=all_ops
    )
    for _ in range(100):
        img = transform(img)
    transform.__repr__()


def test_random_crop():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    oheight = random.randint(5, (height - 2) / 2) * 2
    owidth = random.randint(5, (width - 2) / 2) * 2
    img = torch.ones(3, height, width, dtype=torch.uint8)
    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth)),
            transforms.PILToTensor(),
        ]
    )(img)
    assert result.size(1) == oheight
    assert result.size(2) == owidth

    padding = random.randint(1, 20)
    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth), padding=padding),
            transforms.PILToTensor(),
        ]
    )(img)
    assert result.size(1) == oheight
    assert result.size(2) == owidth

    result = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomCrop((height, width)), transforms.PILToTensor()]
    )(img)
    assert result.size(1) == height
    assert result.size(2) == width
    torch.testing.assert_close(result, img)

    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop((height + 1, width + 1), pad_if_needed=True),
            transforms.PILToTensor(),
        ]
    )(img)
    assert result.size(1) == height + 1
    assert result.size(2) == width + 1

    t = transforms.RandomCrop(33)
    img = torch.ones(3, 32, 32)
    with pytest.raises(ValueError, match=r"Required crop size .+ is larger than input image size .+"):
        t(img)


def test_center_crop():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    oheight = random.randint(5, (height - 2) / 2) * 2
    owidth = random.randint(5, (width - 2) / 2) * 2

    img = torch.ones(3, height, width, dtype=torch.uint8)
    oh1 = (height - oheight) // 2
    ow1 = (width - owidth) // 2
    imgnarrow = img[:, oh1 : oh1 + oheight, ow1 : ow1 + owidth]
    imgnarrow.fill_(0)
    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.PILToTensor(),
        ]
    )(img)
    assert result.sum() == 0
    oheight += 1
    owidth += 1
    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.PILToTensor(),
        ]
    )(img)
    sum1 = result.sum()
    assert sum1 > 1
    oheight += 1
    owidth += 1
    result = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.PILToTensor(),
        ]
    )(img)
    sum2 = result.sum()
    assert sum2 > 0
    assert sum2 > sum1


@pytest.mark.parametrize("odd_image_size", (True, False))
@pytest.mark.parametrize("delta", (1, 3, 5))
@pytest.mark.parametrize("delta_width", (-2, -1, 0, 1, 2))
@pytest.mark.parametrize("delta_height", (-2, -1, 0, 1, 2))
def test_center_crop_2(odd_image_size, delta, delta_width, delta_height):
    """Tests when center crop size is larger than image size, along any dimension"""

    # Since height is independent of width, we can ignore images with odd height and even width and vice-versa.
    input_image_size = (random.randint(10, 32) * 2, random.randint(10, 32) * 2)
    if odd_image_size:
        input_image_size = (input_image_size[0] + 1, input_image_size[1] + 1)

    delta_height *= delta
    delta_width *= delta

    img = torch.ones(3, *input_image_size, dtype=torch.uint8)
    crop_size = (input_image_size[0] + delta_height, input_image_size[1] + delta_width)

    # Test both transforms, one with PIL input and one with tensor
    output_pil = transforms.Compose(
        [transforms.ToPILImage(), transforms.CenterCrop(crop_size), transforms.PILToTensor()],
    )(img)
    assert output_pil.size()[1:3] == crop_size

    output_tensor = transforms.CenterCrop(crop_size)(img)
    assert output_tensor.size()[1:3] == crop_size

    # Ensure output for PIL and Tensor are equal
    assert_equal(
        output_tensor,
        output_pil,
        msg=f"image_size: {input_image_size} crop_size: {crop_size}",
    )

    # Check if content in center of both image and cropped output is same.
    center_size = (min(crop_size[0], input_image_size[0]), min(crop_size[1], input_image_size[1]))
    crop_center_tl, input_center_tl = [0, 0], [0, 0]
    for index in range(2):
        if crop_size[index] > input_image_size[index]:
            crop_center_tl[index] = (crop_size[index] - input_image_size[index]) // 2
        else:
            input_center_tl[index] = (input_image_size[index] - crop_size[index]) // 2

    output_center = output_pil[
        :,
        crop_center_tl[0] : crop_center_tl[0] + center_size[0],
        crop_center_tl[1] : crop_center_tl[1] + center_size[1],
    ]

    img_center = img[
        :,
        input_center_tl[0] : input_center_tl[0] + center_size[0],
        input_center_tl[1] : input_center_tl[1] + center_size[1],
    ]

    assert_equal(output_center, img_center)


def test_color_jitter():
    color_jitter = transforms.ColorJitter(2, 2, 2, 0.1)

    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode="RGB")
    x_pil_2 = x_pil.convert("L")

    for _ in range(10):
        y_pil = color_jitter(x_pil)
        assert y_pil.mode == x_pil.mode

        y_pil_2 = color_jitter(x_pil_2)
        assert y_pil_2.mode == x_pil_2.mode

    # Checking if ColorJitter can be printed as string
    color_jitter.__repr__()


@pytest.mark.parametrize("hue", [1, (-1, 1)])
def test_color_jitter_hue_out_of_bounds(hue):
    with pytest.raises(ValueError, match=re.escape("hue values should be between (-0.5, 0.5)")):
        transforms.ColorJitter(hue=hue)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.skipif(stats is None, reason="scipy.stats not available")
def test_random_erasing(seed):
    torch.random.manual_seed(seed)
    img = torch.ones(3, 128, 128)

    t = transforms.RandomErasing(scale=(0.1, 0.1), ratio=(1 / 3, 3.0))
    y, x, h, w, v = t.get_params(
        img,
        t.scale,
        t.ratio,
        [
            t.value,
        ],
    )
    aspect_ratio = h / w
    # Add some tolerance due to the rounding and int conversion used in the transform
    tol = 0.05
    assert 1 / 3 - tol <= aspect_ratio <= 3 + tol

    # Make sure that h > w and h < w are equally likely (log-scale sampling)
    aspect_ratios = []
    random.seed(42)
    trial = 1000
    for _ in range(trial):
        y, x, h, w, v = t.get_params(
            img,
            t.scale,
            t.ratio,
            [
                t.value,
            ],
        )
        aspect_ratios.append(h / w)

    count_bigger_then_ones = len([1 for aspect_ratio in aspect_ratios if aspect_ratio > 1])
    p_value = stats.binomtest(count_bigger_then_ones, trial, p=0.5).pvalue
    assert p_value > 0.0001

    # Checking if RandomErasing can be printed as string
    t.__repr__()


def test_random_rotation():

    with pytest.raises(ValueError):
        transforms.RandomRotation(-0.7)

    with pytest.raises(ValueError):
        transforms.RandomRotation([-0.7])

    with pytest.raises(ValueError):
        transforms.RandomRotation([-0.7, 0, 0.7])

    t = transforms.RandomRotation(0, fill=None)
    assert t.fill == 0

    t = transforms.RandomRotation(10)
    angle = t.get_params(t.degrees)
    assert angle > -10 and angle < 10

    t = transforms.RandomRotation((-10, 10))
    angle = t.get_params(t.degrees)
    assert -10 < angle < 10

    # Checking if RandomRotation can be printed as string
    t.__repr__()

    t = transforms.RandomRotation((-10, 10), interpolation=Image.BILINEAR)
    assert t.interpolation == transforms.InterpolationMode.BILINEAR


def test_random_rotation_error():
    # assert fill being either a Sequence or a Number
    with pytest.raises(TypeError):
        transforms.RandomRotation(0, fill={})


def test_randomperspective():
    for _ in range(10):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        img = torch.ones(3, height, width)
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(img)
        perp = transforms.RandomPerspective()
        startpoints, endpoints = perp.get_params(width, height, 0.5)
        tr_img = F.perspective(img, startpoints, endpoints)
        tr_img2 = F.convert_image_dtype(F.pil_to_tensor(F.perspective(tr_img, endpoints, startpoints)))
        tr_img = F.convert_image_dtype(F.pil_to_tensor(tr_img))
        assert img.size[0] == width
        assert img.size[1] == height
        assert torch.nn.functional.mse_loss(
            tr_img, F.convert_image_dtype(F.pil_to_tensor(img))
        ) + 0.3 > torch.nn.functional.mse_loss(tr_img2, F.convert_image_dtype(F.pil_to_tensor(img)))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mode", ["L", "RGB", "F"])
def test_randomperspective_fill(mode, seed):
    torch.random.manual_seed(seed)

    # assert fill being either a Sequence or a Number
    with pytest.raises(TypeError):
        transforms.RandomPerspective(fill={})

    t = transforms.RandomPerspective(fill=None)
    assert t.fill == 0

    height = 100
    width = 100
    img = torch.ones(3, height, width)
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(img)
    fill = 127
    num_bands = len(mode)

    img_conv = img.convert(mode)
    perspective = transforms.RandomPerspective(p=1, fill=fill)
    tr_img = perspective(img_conv)
    pixel = tr_img.getpixel((0, 0))

    if not isinstance(pixel, tuple):
        pixel = (pixel,)
    assert pixel == tuple([fill] * num_bands)

    startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.5)
    tr_img = F.perspective(img_conv, startpoints, endpoints, fill=fill)
    pixel = tr_img.getpixel((0, 0))

    if not isinstance(pixel, tuple):
        pixel = (pixel,)
    assert pixel == tuple([fill] * num_bands)

    wrong_num_bands = num_bands + 1
    with pytest.raises(ValueError):
        F.perspective(img_conv, startpoints, endpoints, fill=tuple([fill] * wrong_num_bands))


@pytest.mark.skipif(stats is None, reason="scipy.stats not available")
def test_normalize():
    def samples_from_standard_normal(tensor):
        p_value = stats.kstest(list(tensor.view(-1)), "norm", args=(0, 1)).pvalue
        return p_value > 0.0001

    random_state = random.getstate()
    random.seed(42)
    for channels in [1, 3]:
        img = torch.rand(channels, 10, 10)
        mean = [img[c].mean() for c in range(channels)]
        std = [img[c].std() for c in range(channels)]
        normalized = transforms.Normalize(mean, std)(img)
        assert samples_from_standard_normal(normalized)
    random.setstate(random_state)

    # Checking if Normalize can be printed as string
    transforms.Normalize(mean, std).__repr__()

    # Checking the optional in-place behaviour
    tensor = torch.rand((1, 16, 16))
    tensor_inplace = transforms.Normalize((0.5,), (0.5,), inplace=True)(tensor)
    assert_equal(tensor, tensor_inplace)


@pytest.mark.parametrize("dtype1", [torch.float32, torch.float64])
@pytest.mark.parametrize("dtype2", [torch.int64, torch.float32, torch.float64])
def test_normalize_different_dtype(dtype1, dtype2):
    img = torch.rand(3, 10, 10, dtype=dtype1)
    mean = torch.tensor([1, 2, 3], dtype=dtype2)
    std = torch.tensor([1, 2, 1], dtype=dtype2)
    # checks that it doesn't crash
    transforms.functional.normalize(img, mean, std)


def test_normalize_3d_tensor():
    torch.manual_seed(28)
    n_channels = 3
    img_size = 10
    mean = torch.rand(n_channels)
    std = torch.rand(n_channels)
    img = torch.rand(n_channels, img_size, img_size)
    target = F.normalize(img, mean, std)

    mean_unsqueezed = mean.view(-1, 1, 1)
    std_unsqueezed = std.view(-1, 1, 1)
    result1 = F.normalize(img, mean_unsqueezed, std_unsqueezed)
    result2 = F.normalize(
        img, mean_unsqueezed.repeat(1, img_size, img_size), std_unsqueezed.repeat(1, img_size, img_size)
    )
    torch.testing.assert_close(target, result1)
    torch.testing.assert_close(target, result2)


class TestAffine:
    @pytest.fixture(scope="class")
    def input_img(self):
        input_img = np.zeros((40, 40, 3), dtype=np.uint8)
        for pt in [(16, 16), (20, 16), (20, 20)]:
            for i in range(-5, 5):
                for j in range(-5, 5):
                    input_img[pt[0] + i, pt[1] + j, :] = [255, 155, 55]
        return input_img

    def test_affine_translate_seq(self, input_img):
        with pytest.raises(TypeError, match=r"Argument translate should be a sequence"):
            F.affine(input_img, 10, translate=0, scale=1, shear=1)

    @pytest.fixture(scope="class")
    def pil_image(self, input_img):
        return F.to_pil_image(input_img)

    def _to_3x3_inv(self, inv_result_matrix):
        result_matrix = np.zeros((3, 3))
        result_matrix[:2, :] = np.array(inv_result_matrix).reshape((2, 3))
        result_matrix[2, 2] = 1
        return np.linalg.inv(result_matrix)

    def _test_transformation(self, angle, translate, scale, shear, pil_image, input_img, center=None):

        a_rad = math.radians(angle)
        s_rad = [math.radians(sh_) for sh_ in shear]
        cnt = [20, 20] if center is None else center
        cx, cy = cnt
        tx, ty = translate
        sx, sy = s_rad
        rot = a_rad

        # 1) Check transformation matrix:
        C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        Cinv = np.linalg.inv(C)

        RS = np.array(
            [
                [scale * math.cos(rot), -scale * math.sin(rot), 0],
                [scale * math.sin(rot), scale * math.cos(rot), 0],
                [0, 0, 1],
            ]
        )

        SHx = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])

        SHy = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])

        RSS = np.matmul(RS, np.matmul(SHy, SHx))

        true_matrix = np.matmul(T, np.matmul(C, np.matmul(RSS, Cinv)))

        result_matrix = self._to_3x3_inv(
            F._get_inverse_affine_matrix(center=cnt, angle=angle, translate=translate, scale=scale, shear=shear)
        )
        assert np.sum(np.abs(true_matrix - result_matrix)) < 1e-10
        # 2) Perform inverse mapping:
        true_result = np.zeros((40, 40, 3), dtype=np.uint8)
        inv_true_matrix = np.linalg.inv(true_matrix)
        for y in range(true_result.shape[0]):
            for x in range(true_result.shape[1]):
                # Same as for PIL:
                # https://github.com/python-pillow/Pillow/blob/71f8ec6a0cfc1008076a023c0756542539d057ab/
                # src/libImaging/Geometry.c#L1060
                input_pt = np.array([x + 0.5, y + 0.5, 1.0])
                res = np.floor(np.dot(inv_true_matrix, input_pt)).astype(int)
                _x, _y = res[:2]
                if 0 <= _x < input_img.shape[1] and 0 <= _y < input_img.shape[0]:
                    true_result[y, x, :] = input_img[_y, _x, :]

        result = F.affine(pil_image, angle=angle, translate=translate, scale=scale, shear=shear, center=center)
        assert result.size == pil_image.size
        # Compute number of different pixels:
        np_result = np.array(result)
        n_diff_pixels = np.sum(np_result != true_result) / 3
        # Accept 3 wrong pixels
        error_msg = (
            f"angle={angle}, translate={translate}, scale={scale}, shear={shear}\nn diff pixels={n_diff_pixels}\n"
        )
        assert n_diff_pixels < 3, error_msg

    def test_transformation_discrete(self, pil_image, input_img):
        # Test rotation
        angle = 45
        self._test_transformation(
            angle=angle, translate=(0, 0), scale=1.0, shear=(0.0, 0.0), pil_image=pil_image, input_img=input_img
        )

        # Test rotation
        angle = 45
        self._test_transformation(
            angle=angle,
            translate=(0, 0),
            scale=1.0,
            shear=(0.0, 0.0),
            pil_image=pil_image,
            input_img=input_img,
            center=[0, 0],
        )

        # Test translation
        translate = [10, 15]
        self._test_transformation(
            angle=0.0, translate=translate, scale=1.0, shear=(0.0, 0.0), pil_image=pil_image, input_img=input_img
        )

        # Test scale
        scale = 1.2
        self._test_transformation(
            angle=0.0, translate=(0.0, 0.0), scale=scale, shear=(0.0, 0.0), pil_image=pil_image, input_img=input_img
        )

        # Test shear
        shear = [45.0, 25.0]
        self._test_transformation(
            angle=0.0, translate=(0.0, 0.0), scale=1.0, shear=shear, pil_image=pil_image, input_img=input_img
        )

        # Test shear with top-left as center
        shear = [45.0, 25.0]
        self._test_transformation(
            angle=0.0,
            translate=(0.0, 0.0),
            scale=1.0,
            shear=shear,
            pil_image=pil_image,
            input_img=input_img,
            center=[0, 0],
        )

    @pytest.mark.parametrize("angle", range(-90, 90, 36))
    @pytest.mark.parametrize("translate", range(-10, 10, 5))
    @pytest.mark.parametrize("scale", [0.77, 1.0, 1.27])
    @pytest.mark.parametrize("shear", range(-15, 15, 5))
    def test_transformation_range(self, angle, translate, scale, shear, pil_image, input_img):
        self._test_transformation(
            angle=angle,
            translate=(translate, translate),
            scale=scale,
            shear=(shear, shear),
            pil_image=pil_image,
            input_img=input_img,
        )


def test_random_affine():

    with pytest.raises(ValueError):
        transforms.RandomAffine(-0.7)
    with pytest.raises(ValueError):
        transforms.RandomAffine([-0.7])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-0.7, 0, 0.7])
    with pytest.raises(TypeError):
        transforms.RandomAffine([-90, 90], translate=2.0)
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[-1.0, 1.0])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[-1.0, 0.0, 1.0])

    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.0])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[-1.0, 1.0])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, -0.5])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 3.0, -0.5])

    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=-7)
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10])
    with pytest.raises(ValueError):
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10, 0, 10])

    # assert fill being either a Sequence or a Number
    with pytest.raises(TypeError):
        transforms.RandomAffine(0, fill={})

    t = transforms.RandomAffine(0, fill=None)
    assert t.fill == 0

    x = np.zeros((100, 100, 3), dtype=np.uint8)
    img = F.to_pil_image(x)

    t = transforms.RandomAffine(10, translate=[0.5, 0.3], scale=[0.7, 1.3], shear=[-10, 10, 20, 40])
    for _ in range(100):
        angle, translations, scale, shear = t.get_params(t.degrees, t.translate, t.scale, t.shear, img_size=img.size)
        assert -10 < angle < 10
        assert -img.size[0] * 0.5 <= translations[0] <= img.size[0] * 0.5
        assert -img.size[1] * 0.5 <= translations[1] <= img.size[1] * 0.5
        assert 0.7 < scale < 1.3
        assert -10 < shear[0] < 10
        assert -20 < shear[1] < 40

    # Checking if RandomAffine can be printed as string
    t.__repr__()

    t = transforms.RandomAffine(10, interpolation=transforms.InterpolationMode.BILINEAR)
    assert "bilinear" in t.__repr__()

    t = transforms.RandomAffine(10, interpolation=Image.BILINEAR)
    assert t.interpolation == transforms.InterpolationMode.BILINEAR


def test_elastic_transformation():
    with pytest.raises(TypeError, match=r"alpha should be float or a sequence of floats"):
        transforms.ElasticTransform(alpha=True, sigma=2.0)
    with pytest.raises(TypeError, match=r"alpha should be a sequence of floats"):
        transforms.ElasticTransform(alpha=[1.0, True], sigma=2.0)
    with pytest.raises(ValueError, match=r"alpha is a sequence its length should be 2"):
        transforms.ElasticTransform(alpha=[1.0, 0.0, 1.0], sigma=2.0)

    with pytest.raises(TypeError, match=r"sigma should be float or a sequence of floats"):
        transforms.ElasticTransform(alpha=2.0, sigma=True)
    with pytest.raises(TypeError, match=r"sigma should be a sequence of floats"):
        transforms.ElasticTransform(alpha=2.0, sigma=[1.0, True])
    with pytest.raises(ValueError, match=r"sigma is a sequence its length should be 2"):
        transforms.ElasticTransform(alpha=2.0, sigma=[1.0, 0.0, 1.0])

    t = transforms.transforms.ElasticTransform(alpha=2.0, sigma=2.0, interpolation=Image.BILINEAR)
    assert t.interpolation == transforms.InterpolationMode.BILINEAR

    with pytest.raises(TypeError, match=r"fill should be int or float"):
        transforms.ElasticTransform(alpha=1.0, sigma=1.0, fill={})

    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    img = F.to_pil_image(x)
    t = transforms.ElasticTransform(alpha=0.0, sigma=0.0)
    transformed_img = t(img)
    assert transformed_img == img

    # Smoke test on PIL images
    t = transforms.ElasticTransform(alpha=0.5, sigma=0.23)
    transformed_img = t(img)
    assert isinstance(transformed_img, Image.Image)

    # Checking if ElasticTransform can be printed as string
    t.__repr__()


def test_random_grayscale_with_grayscale_input():
    transform = transforms.RandomGrayscale(p=1.0)

    image_tensor = torch.randint(0, 256, (1, 16, 16), dtype=torch.uint8)
    output_tensor = transform(image_tensor)
    torch.testing.assert_close(output_tensor, image_tensor)

    image_pil = F.to_pil_image(image_tensor)
    output_pil = transform(image_pil)
    torch.testing.assert_close(F.pil_to_tensor(output_pil), image_tensor)


if __name__ == "__main__":
    pytest.main([__file__])
