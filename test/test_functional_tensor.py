import colorsys
import itertools
import math
import os
from functools import partial
from typing import Sequence

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision.transforms as T
import torchvision.transforms._functional_pil as F_pil
import torchvision.transforms._functional_tensor as F_t
import torchvision.transforms.functional as F
from common_utils import (
    _assert_approx_equal_tensor_to_pil,
    _assert_equal_tensor_to_pil,
    _create_data,
    _create_data_batch,
    _test_fn_on_batch,
    assert_equal,
    cpu_and_cuda,
    needs_cuda,
)
from torchvision.transforms import InterpolationMode

NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC = (
    InterpolationMode.NEAREST,
    InterpolationMode.NEAREST_EXACT,
    InterpolationMode.BILINEAR,
    InterpolationMode.BICUBIC,
)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("fn", [F.get_image_size, F.get_image_num_channels, F.get_dimensions])
def test_image_sizes(device, fn):
    script_F = torch.jit.script(fn)

    img_tensor, pil_img = _create_data(16, 18, 3, device=device)
    value_img = fn(img_tensor)
    value_pil_img = fn(pil_img)
    assert value_img == value_pil_img

    value_img_script = script_F(img_tensor)
    assert value_img == value_img_script

    batch_tensors = _create_data_batch(16, 18, 3, num_samples=4, device=device)
    value_img_batch = fn(batch_tensors)
    assert value_img == value_img_batch


@needs_cuda
def test_scale_channel():
    """Make sure that _scale_channel gives the same results on CPU and GPU as
    histc or bincount are used depending on the device.
    """
    # TODO: when # https://github.com/pytorch/pytorch/issues/53194 is fixed,
    # only use bincount and remove that test.
    size = (1_000,)
    img_chan = torch.randint(0, 256, size=size).to("cpu")
    scaled_cpu = F_t._scale_channel(img_chan)
    scaled_cuda = F_t._scale_channel(img_chan.to("cuda"))
    assert_equal(scaled_cpu, scaled_cuda.to("cpu"))


class TestRotate:

    ALL_DTYPES = [None, torch.float32, torch.float64, torch.float16]
    scripted_rotate = torch.jit.script(F.rotate)
    IMG_W = 26

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(7, 33), (26, IMG_W), (32, IMG_W)])
    @pytest.mark.parametrize(
        "center",
        [
            None,
            (int(IMG_W * 0.3), int(IMG_W * 0.4)),
            [int(IMG_W * 0.5), int(IMG_W * 0.6)],
        ],
    )
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    @pytest.mark.parametrize("angle", range(-180, 180, 34))
    @pytest.mark.parametrize("expand", [True, False])
    @pytest.mark.parametrize(
        "fill",
        [
            None,
            [0, 0, 0],
            (1, 2, 3),
            [255, 255, 255],
            [
                1,
            ],
            (2.0,),
        ],
    )
    @pytest.mark.parametrize("fn", [F.rotate, scripted_rotate])
    def test_rotate(self, device, height, width, center, dt, angle, expand, fill, fn):
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and torch.device(device).type == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        f_pil = int(fill[0]) if fill is not None and len(fill) == 1 else fill
        out_pil_img = F.rotate(pil_img, angle=angle, interpolation=NEAREST, expand=expand, center=center, fill=f_pil)
        out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

        out_tensor = fn(tensor, angle=angle, interpolation=NEAREST, expand=expand, center=center, fill=fill).cpu()

        if out_tensor.dtype != torch.uint8:
            out_tensor = out_tensor.to(torch.uint8)

        assert (
            out_tensor.shape == out_pil_tensor.shape
        ), f"{(height, width, NEAREST, dt, angle, expand, center)}: {out_tensor.shape} vs {out_pil_tensor.shape}"

        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
        # Tolerance : less than 3% of different pixels
        assert ratio_diff_pixels < 0.03, (
            f"{(height, width, NEAREST, dt, angle, expand, center, fill)}: "
            f"{ratio_diff_pixels}\n{out_tensor[0, :7, :7]} vs \n"
            f"{out_pil_tensor[0, :7, :7]}"
        )

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    def test_rotate_batch(self, device, dt):
        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        batch_tensors = _create_data_batch(26, 36, num_samples=4, device=device)
        if dt is not None:
            batch_tensors = batch_tensors.to(dtype=dt)

        center = (20, 22)
        _test_fn_on_batch(batch_tensors, F.rotate, angle=32, interpolation=NEAREST, expand=True, center=center)

    def test_rotate_interpolation_type(self):
        tensor, _ = _create_data(26, 26)
        res1 = F.rotate(tensor, 45, interpolation=PIL.Image.BILINEAR)
        res2 = F.rotate(tensor, 45, interpolation=BILINEAR)
        assert_equal(res1, res2)


class TestAffine:

    ALL_DTYPES = [None, torch.float32, torch.float64, torch.float16]
    scripted_affine = torch.jit.script(F.affine)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(26, 26), (32, 26)])
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    def test_identity_map(self, device, height, width, dt):
        # Tests on square and rectangular images
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        # 1) identity map
        out_tensor = F.affine(tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST)

        assert_equal(tensor, out_tensor, msg=f"{out_tensor[0, :5, :5]} vs {tensor[0, :5, :5]}")
        out_tensor = self.scripted_affine(
            tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST
        )
        assert_equal(tensor, out_tensor, msg=f"{out_tensor[0, :5, :5]} vs {tensor[0, :5, :5]}")

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(26, 26)])
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    @pytest.mark.parametrize(
        "angle, config",
        [
            (90, {"k": 1, "dims": (-1, -2)}),
            (45, None),
            (30, None),
            (-30, None),
            (-45, None),
            (-90, {"k": -1, "dims": (-1, -2)}),
            (180, {"k": 2, "dims": (-1, -2)}),
        ],
    )
    @pytest.mark.parametrize("fn", [F.affine, scripted_affine])
    def test_square_rotations(self, device, height, width, dt, angle, config, fn):
        # 2) Test rotation
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        out_pil_img = F.affine(
            pil_img, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST
        )
        out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1))).to(device)

        out_tensor = fn(tensor, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST)
        if config is not None:
            assert_equal(torch.rot90(tensor, **config), out_tensor)

        if out_tensor.dtype != torch.uint8:
            out_tensor = out_tensor.to(torch.uint8)

        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
        # Tolerance : less than 6% of different pixels
        assert ratio_diff_pixels < 0.06

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(32, 26)])
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    @pytest.mark.parametrize("angle", [90, 45, 15, -30, -60, -120])
    @pytest.mark.parametrize("fn", [F.affine, scripted_affine])
    @pytest.mark.parametrize("center", [None, [0, 0]])
    def test_rect_rotations(self, device, height, width, dt, angle, fn, center):
        # Tests on rectangular images
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        out_pil_img = F.affine(
            pil_img, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST, center=center
        )
        out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

        out_tensor = fn(
            tensor, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST, center=center
        ).cpu()

        if out_tensor.dtype != torch.uint8:
            out_tensor = out_tensor.to(torch.uint8)

        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
        # Tolerance : less than 3% of different pixels
        assert ratio_diff_pixels < 0.03

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(26, 26), (32, 26)])
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    @pytest.mark.parametrize("t", [[10, 12], (-12, -13)])
    @pytest.mark.parametrize("fn", [F.affine, scripted_affine])
    def test_translations(self, device, height, width, dt, t, fn):
        # 3) Test translation
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        out_pil_img = F.affine(pil_img, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST)

        out_tensor = fn(tensor, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], interpolation=NEAREST)

        if out_tensor.dtype != torch.uint8:
            out_tensor = out_tensor.to(torch.uint8)

        _assert_equal_tensor_to_pil(out_tensor, out_pil_img)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("height, width", [(26, 26), (32, 26)])
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    @pytest.mark.parametrize(
        "a, t, s, sh, f",
        [
            (45.5, [5, 6], 1.0, [0.0, 0.0], None),
            (33, (5, -4), 1.0, [0.0, 0.0], [0, 0, 0]),
            (45, [-5, 4], 1.2, [0.0, 0.0], (1, 2, 3)),
            (33, (-4, -8), 2.0, [0.0, 0.0], [255, 255, 255]),
            (85, (10, -10), 0.7, [0.0, 0.0], [1]),
            (0, [0, 0], 1.0, [35.0], (2.0,)),
            (-25, [0, 0], 1.2, [0.0, 15.0], None),
            (-45, [-10, 0], 0.7, [2.0, 5.0], None),
            (-45, [-10, -10], 1.2, [4.0, 5.0], None),
            (-90, [0, 0], 1.0, [0.0, 0.0], None),
        ],
    )
    @pytest.mark.parametrize("fn", [F.affine, scripted_affine])
    def test_all_ops(self, device, height, width, dt, a, t, s, sh, f, fn):
        # 4) Test rotation + translation + scale + shear
        tensor, pil_img = _create_data(height, width, device=device)

        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        if dt is not None:
            tensor = tensor.to(dtype=dt)

        f_pil = int(f[0]) if f is not None and len(f) == 1 else f
        out_pil_img = F.affine(pil_img, angle=a, translate=t, scale=s, shear=sh, interpolation=NEAREST, fill=f_pil)
        out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

        out_tensor = fn(tensor, angle=a, translate=t, scale=s, shear=sh, interpolation=NEAREST, fill=f).cpu()

        if out_tensor.dtype != torch.uint8:
            out_tensor = out_tensor.to(torch.uint8)

        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
        # Tolerance : less than 5% (cpu), 6% (cuda) of different pixels
        tol = 0.06 if device == "cuda" else 0.05
        assert ratio_diff_pixels < tol

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dt", ALL_DTYPES)
    def test_batches(self, device, dt):
        if dt == torch.float16 and device == "cpu":
            # skip float16 on CPU case
            return

        batch_tensors = _create_data_batch(26, 36, num_samples=4, device=device)
        if dt is not None:
            batch_tensors = batch_tensors.to(dtype=dt)

        _test_fn_on_batch(batch_tensors, F.affine, angle=-43, translate=[-3, 4], scale=1.2, shear=[4.0, 5.0])

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_interpolation_type(self, device):
        tensor, pil_img = _create_data(26, 26, device=device)

        res1 = F.affine(tensor, 45, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=PIL.Image.BILINEAR)
        res2 = F.affine(tensor, 45, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=BILINEAR)
        assert_equal(res1, res2)


def _get_data_dims_and_points_for_perspective():
    # Ideally we would parametrize independently over data dims and points, but
    # we want to tests on some points that also depend on the data dims.
    # Pytest doesn't support covariant parametrization, so we do it somewhat manually here.

    data_dims = [(26, 34), (26, 26)]
    points = [
        [[[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]],
        [[[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]],
    ]

    dims_and_points = list(itertools.product(data_dims, points))

    # up to here, we could just have used 2 @parametrized.
    # Down below is the covarariant part as the points depend on the data dims.

    n = 10
    for dim in data_dims:
        points += [(dim, T.RandomPerspective.get_params(dim[1], dim[0], i / n)) for i in range(n)]
    return dims_and_points


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dims_and_points", _get_data_dims_and_points_for_perspective())
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("fill", (None, [0, 0, 0], [1, 2, 3], [255, 255, 255], [1], (2.0,)))
@pytest.mark.parametrize("fn", [F.perspective, torch.jit.script(F.perspective)])
def test_perspective_pil_vs_tensor(device, dims_and_points, dt, fill, fn):

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    data_dims, (spoints, epoints) = dims_and_points

    tensor, pil_img = _create_data(*data_dims, device=device)
    if dt is not None:
        tensor = tensor.to(dtype=dt)

    interpolation = NEAREST
    fill_pil = int(fill[0]) if fill is not None and len(fill) == 1 else fill
    out_pil_img = F.perspective(
        pil_img, startpoints=spoints, endpoints=epoints, interpolation=interpolation, fill=fill_pil
    )
    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))
    out_tensor = fn(tensor, startpoints=spoints, endpoints=epoints, interpolation=interpolation, fill=fill).cpu()

    if out_tensor.dtype != torch.uint8:
        out_tensor = out_tensor.to(torch.uint8)

    num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
    ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
    # Tolerance : less than 5% of different pixels
    assert ratio_diff_pixels < 0.05


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dims_and_points", _get_data_dims_and_points_for_perspective())
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
def test_perspective_batch(device, dims_and_points, dt):

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    data_dims, (spoints, epoints) = dims_and_points

    batch_tensors = _create_data_batch(*data_dims, num_samples=4, device=device)
    if dt is not None:
        batch_tensors = batch_tensors.to(dtype=dt)

    # Ignore the equivalence between scripted and regular function on float16 cuda. The pixels at
    # the border may be entirely different due to small rounding errors.
    scripted_fn_atol = -1 if (dt == torch.float16 and device == "cuda") else 1e-8
    _test_fn_on_batch(
        batch_tensors,
        F.perspective,
        scripted_fn_atol=scripted_fn_atol,
        startpoints=spoints,
        endpoints=epoints,
        interpolation=NEAREST,
    )


def test_perspective_interpolation_type():
    spoints = [[0, 0], [33, 0], [33, 25], [0, 25]]
    epoints = [[3, 2], [32, 3], [30, 24], [2, 25]]
    tensor = torch.randint(0, 256, (3, 26, 26))

    res1 = F.perspective(tensor, startpoints=spoints, endpoints=epoints, interpolation=PIL.Image.BILINEAR)
    res2 = F.perspective(tensor, startpoints=spoints, endpoints=epoints, interpolation=BILINEAR)
    assert_equal(res1, res2)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("size", [32, 26, [32], [32, 32], (32, 32), [26, 35]])
@pytest.mark.parametrize("max_size", [None, 34, 40, 1000])
@pytest.mark.parametrize("interpolation", [BILINEAR, BICUBIC, NEAREST, NEAREST_EXACT])
def test_resize(device, dt, size, max_size, interpolation):

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    if max_size is not None and isinstance(size, Sequence) and len(size) != 1:
        return  # unsupported

    torch.manual_seed(12)
    script_fn = torch.jit.script(F.resize)
    tensor, pil_img = _create_data(26, 36, device=device)
    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)

    if dt is not None:
        # This is a trivial cast to float of uint8 data to test all cases
        tensor = tensor.to(dt)
        batch_tensors = batch_tensors.to(dt)

    resized_tensor = F.resize(tensor, size=size, interpolation=interpolation, max_size=max_size, antialias=True)
    resized_pil_img = F.resize(pil_img, size=size, interpolation=interpolation, max_size=max_size, antialias=True)

    assert resized_tensor.size()[1:] == resized_pil_img.size[::-1]

    if interpolation != NEAREST:
        # We can not check values if mode = NEAREST, as results are different
        # E.g. resized_tensor  = [[a, a, b, c, d, d, e, ...]]
        # E.g. resized_pil_img = [[a, b, c, c, d, e, f, ...]]
        resized_tensor_f = resized_tensor
        # we need to cast to uint8 to compare with PIL image
        if resized_tensor_f.dtype == torch.uint8:
            resized_tensor_f = resized_tensor_f.to(torch.float)

        # Pay attention to high tolerance for MAE
        _assert_approx_equal_tensor_to_pil(resized_tensor_f, resized_pil_img, tol=3.0)

    if isinstance(size, int):
        script_size = [size]
    else:
        script_size = size

    resize_result = script_fn(tensor, size=script_size, interpolation=interpolation, max_size=max_size, antialias=True)
    assert_equal(resized_tensor, resize_result)

    _test_fn_on_batch(
        batch_tensors, F.resize, size=script_size, interpolation=interpolation, max_size=max_size, antialias=True
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_resize_asserts(device):

    tensor, pil_img = _create_data(26, 36, device=device)

    res1 = F.resize(tensor, size=32, interpolation=PIL.Image.BILINEAR)
    res2 = F.resize(tensor, size=32, interpolation=BILINEAR)
    assert_equal(res1, res2)

    for img in (tensor, pil_img):
        exp_msg = "max_size should only be passed if size specifies the length of the smaller edge"
        with pytest.raises(ValueError, match=exp_msg):
            F.resize(img, size=(32, 34), max_size=35)
        with pytest.raises(ValueError, match="max_size = 32 must be strictly greater"):
            F.resize(img, size=32, max_size=32)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("size", [[96, 72], [96, 420], [420, 72]])
@pytest.mark.parametrize("interpolation", [BILINEAR, BICUBIC])
def test_resize_antialias(device, dt, size, interpolation):

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    torch.manual_seed(12)
    script_fn = torch.jit.script(F.resize)
    tensor, pil_img = _create_data(320, 290, device=device)

    if dt is not None:
        # This is a trivial cast to float of uint8 data to test all cases
        tensor = tensor.to(dt)

    resized_tensor = F.resize(tensor, size=size, interpolation=interpolation, antialias=True)
    resized_pil_img = F.resize(pil_img, size=size, interpolation=interpolation, antialias=True)

    assert resized_tensor.size()[1:] == resized_pil_img.size[::-1]

    resized_tensor_f = resized_tensor
    # we need to cast to uint8 to compare with PIL image
    if resized_tensor_f.dtype == torch.uint8:
        resized_tensor_f = resized_tensor_f.to(torch.float)

    _assert_approx_equal_tensor_to_pil(resized_tensor_f, resized_pil_img, tol=0.5, msg=f"{size}, {interpolation}, {dt}")

    accepted_tol = 1.0 + 1e-5
    if interpolation == BICUBIC:
        # this overall mean value to make the tests pass
        # High value is mostly required for test cases with
        # downsampling and upsampling where we can not exactly
        # match PIL implementation.
        accepted_tol = 15.0

    _assert_approx_equal_tensor_to_pil(
        resized_tensor_f, resized_pil_img, tol=accepted_tol, agg_method="max", msg=f"{size}, {interpolation}, {dt}"
    )

    if isinstance(size, int):
        script_size = [
            size,
        ]
    else:
        script_size = size

    resize_result = script_fn(tensor, size=script_size, interpolation=interpolation, antialias=True)
    assert_equal(resized_tensor, resize_result)


def check_functional_vs_PIL_vs_scripted(
    fn, fn_pil, fn_t, config, device, dtype, channels=3, tol=2.0 + 1e-10, agg_method="max"
):

    script_fn = torch.jit.script(fn)
    torch.manual_seed(15)
    tensor, pil_img = _create_data(26, 34, channels=channels, device=device)
    batch_tensors = _create_data_batch(16, 18, num_samples=4, channels=channels, device=device)

    if dtype is not None:
        tensor = F.convert_image_dtype(tensor, dtype)
        batch_tensors = F.convert_image_dtype(batch_tensors, dtype)

    out_fn_t = fn_t(tensor, **config)
    out_pil = fn_pil(pil_img, **config)
    out_scripted = script_fn(tensor, **config)
    assert out_fn_t.dtype == out_scripted.dtype
    assert out_fn_t.size()[1:] == out_pil.size[::-1]

    rbg_tensor = out_fn_t

    if out_fn_t.dtype != torch.uint8:
        rbg_tensor = F.convert_image_dtype(out_fn_t, torch.uint8)

    # Check that max difference does not exceed 2 in [0, 255] range
    # Exact matching is not possible due to incompatibility convert_image_dtype and PIL results
    _assert_approx_equal_tensor_to_pil(rbg_tensor.float(), out_pil, tol=tol, agg_method=agg_method)

    atol = 1e-6
    if out_fn_t.dtype == torch.uint8 and "cuda" in torch.device(device).type:
        atol = 1.0
    assert out_fn_t.allclose(out_scripted, atol=atol)

    # FIXME: fn will be scripted again in _test_fn_on_batch. We could avoid that.
    _test_fn_on_batch(batch_tensors, fn, scripted_fn_atol=atol, **config)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"brightness_factor": f} for f in (0.1, 0.5, 1.0, 1.34, 2.5)])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_brightness(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_brightness,
        F_pil.adjust_brightness,
        F_t.adjust_brightness,
        config,
        device,
        dtype,
        channels,
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("channels", [1, 3])
def test_invert(device, dtype, channels):
    check_functional_vs_PIL_vs_scripted(
        F.invert, F_pil.invert, F_t.invert, {}, device, dtype, channels, tol=1.0, agg_method="max"
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("config", [{"bits": bits} for bits in range(0, 8)])
@pytest.mark.parametrize("channels", [1, 3])
def test_posterize(device, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.posterize,
        F_pil.posterize,
        F_t.posterize,
        config,
        device,
        dtype=None,
        channels=channels,
        tol=1.0,
        agg_method="max",
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("config", [{"threshold": threshold} for threshold in [0, 64, 128, 192, 255]])
@pytest.mark.parametrize("channels", [1, 3])
def test_solarize1(device, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.solarize,
        F_pil.solarize,
        F_t.solarize,
        config,
        device,
        dtype=None,
        channels=channels,
        tol=1.0,
        agg_method="max",
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"threshold": threshold} for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]])
@pytest.mark.parametrize("channels", [1, 3])
def test_solarize2(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.solarize,
        lambda img, threshold: F_pil.solarize(img, 255 * threshold),
        F_t.solarize,
        config,
        device,
        dtype,
        channels,
        tol=1.0,
        agg_method="max",
    )


@pytest.mark.parametrize(
    ("dtype", "threshold"),
    [
        *[
            (dtype, threshold)
            for dtype, threshold in itertools.product(
                [torch.float32, torch.float16],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
        ],
        *[(torch.uint8, threshold) for threshold in [0, 64, 128, 192, 255]],
        *[(torch.int64, threshold) for threshold in [0, 2**32, 2**63 - 1]],
    ],
)
@pytest.mark.parametrize("device", cpu_and_cuda())
def test_solarize_threshold_within_bound(threshold, dtype, device):
    make_img = torch.rand if dtype.is_floating_point else partial(torch.randint, 0, torch.iinfo(dtype).max)
    img = make_img((3, 12, 23), dtype=dtype, device=device)
    F_t.solarize(img, threshold)


@pytest.mark.parametrize(
    ("dtype", "threshold"),
    [
        (torch.float32, 1.5),
        (torch.float16, 1.5),
        (torch.uint8, 260),
        (torch.int64, 2**64),
    ],
)
@pytest.mark.parametrize("device", cpu_and_cuda())
def test_solarize_threshold_above_bound(threshold, dtype, device):
    make_img = torch.rand if dtype.is_floating_point else partial(torch.randint, 0, torch.iinfo(dtype).max)
    img = make_img((3, 12, 23), dtype=dtype, device=device)
    with pytest.raises(TypeError, match="Threshold should be less than bound of img."):
        F_t.solarize(img, threshold)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"sharpness_factor": f} for f in [0.2, 0.5, 1.0, 1.5, 2.0]])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_sharpness(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_sharpness,
        F_pil.adjust_sharpness,
        F_t.adjust_sharpness,
        config,
        device,
        dtype,
        channels,
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("channels", [1, 3])
def test_autocontrast(device, dtype, channels):
    check_functional_vs_PIL_vs_scripted(
        F.autocontrast, F_pil.autocontrast, F_t.autocontrast, {}, device, dtype, channels, tol=1.0, agg_method="max"
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("channels", [1, 3])
def test_autocontrast_equal_minmax(device, dtype, channels):
    a = _create_data_batch(32, 32, num_samples=1, channels=channels, device=device)
    a = a / 2.0 + 0.3
    assert (F.autocontrast(a)[0] == F.autocontrast(a[0])).all()

    a[0, 0] = 0.7
    assert (F.autocontrast(a)[0] == F.autocontrast(a[0])).all()


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("channels", [1, 3])
def test_equalize(device, channels):
    torch.use_deterministic_algorithms(False)
    check_functional_vs_PIL_vs_scripted(
        F.equalize,
        F_pil.equalize,
        F_t.equalize,
        {},
        device,
        dtype=None,
        channels=channels,
        tol=1.0,
        agg_method="max",
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"contrast_factor": f} for f in [0.2, 0.5, 1.0, 1.5, 2.0]])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_contrast(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_contrast, F_pil.adjust_contrast, F_t.adjust_contrast, config, device, dtype, channels
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"saturation_factor": f} for f in [0.5, 0.75, 1.0, 1.5, 2.0]])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_saturation(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_saturation, F_pil.adjust_saturation, F_t.adjust_saturation, config, device, dtype, channels
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"hue_factor": f} for f in [-0.45, -0.25, 0.0, 0.25, 0.45]])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_hue(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_hue, F_pil.adjust_hue, F_t.adjust_hue, config, device, dtype, channels, tol=16.1, agg_method="max"
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float64))
@pytest.mark.parametrize("config", [{"gamma": g1, "gain": g2} for g1, g2 in zip([0.8, 1.0, 1.2], [0.7, 1.0, 1.3])])
@pytest.mark.parametrize("channels", [1, 3])
def test_adjust_gamma(device, dtype, config, channels):
    check_functional_vs_PIL_vs_scripted(
        F.adjust_gamma,
        F_pil.adjust_gamma,
        F_t.adjust_gamma,
        config,
        device,
        dtype,
        channels,
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("pad", [2, [3], [0, 3], (3, 3), [4, 2, 4, 3]])
@pytest.mark.parametrize(
    "config",
    [
        {"padding_mode": "constant", "fill": 0},
        {"padding_mode": "constant", "fill": 10},
        {"padding_mode": "constant", "fill": 20.2},
        {"padding_mode": "edge"},
        {"padding_mode": "reflect"},
        {"padding_mode": "symmetric"},
    ],
)
def test_pad(device, dt, pad, config):
    script_fn = torch.jit.script(F.pad)
    tensor, pil_img = _create_data(7, 8, device=device)
    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    if dt is not None:
        # This is a trivial cast to float of uint8 data to test all cases
        tensor = tensor.to(dt)
        batch_tensors = batch_tensors.to(dt)

    pad_tensor = F_t.pad(tensor, pad, **config)
    pad_pil_img = F_pil.pad(pil_img, pad, **config)

    pad_tensor_8b = pad_tensor
    # we need to cast to uint8 to compare with PIL image
    if pad_tensor_8b.dtype != torch.uint8:
        pad_tensor_8b = pad_tensor_8b.to(torch.uint8)

    _assert_equal_tensor_to_pil(pad_tensor_8b, pad_pil_img, msg=f"{pad}, {config}")

    if isinstance(pad, int):
        script_pad = [
            pad,
        ]
    else:
        script_pad = pad
    pad_tensor_script = script_fn(tensor, script_pad, **config)
    assert_equal(pad_tensor, pad_tensor_script, msg=f"{pad}, {config}")

    _test_fn_on_batch(batch_tensors, F.pad, padding=script_pad, **config)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("mode", [NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC])
def test_resized_crop(device, mode):
    # test values of F.resized_crop in several cases:
    # 1) resize to the same size, crop to the same size => should be identity
    tensor, _ = _create_data(26, 36, device=device)

    out_tensor = F.resized_crop(
        tensor, top=0, left=0, height=26, width=36, size=[26, 36], interpolation=mode, antialias=True
    )
    assert_equal(tensor, out_tensor, msg=f"{out_tensor[0, :5, :5]} vs {tensor[0, :5, :5]}")

    # 2) resize by half and crop a TL corner
    tensor, _ = _create_data(26, 36, device=device)
    out_tensor = F.resized_crop(tensor, top=0, left=0, height=20, width=30, size=[10, 15], interpolation=NEAREST)
    expected_out_tensor = tensor[:, :20:2, :30:2]
    assert_equal(
        expected_out_tensor,
        out_tensor,
        msg=f"{expected_out_tensor[0, :10, :10]} vs {out_tensor[0, :10, :10]}",
    )

    batch_tensors = _create_data_batch(26, 36, num_samples=4, device=device)
    _test_fn_on_batch(
        batch_tensors,
        F.resized_crop,
        top=1,
        left=2,
        height=20,
        width=30,
        size=[10, 15],
        interpolation=NEAREST,
    )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "func, args",
    [
        (F_t.get_dimensions, ()),
        (F_t.get_image_size, ()),
        (F_t.get_image_num_channels, ()),
        (F_t.vflip, ()),
        (F_t.hflip, ()),
        (F_t.crop, (1, 2, 4, 5)),
        (F_t.adjust_brightness, (0.0,)),
        (F_t.adjust_contrast, (1.0,)),
        (F_t.adjust_hue, (-0.5,)),
        (F_t.adjust_saturation, (2.0,)),
        (F_t.pad, ([2], 2, "constant")),
        (F_t.resize, ([10, 11],)),
        (F_t.perspective, ([0.2])),
        (F_t.gaussian_blur, ((2, 2), (0.7, 0.5))),
        (F_t.invert, ()),
        (F_t.posterize, (0,)),
        (F_t.solarize, (0.3,)),
        (F_t.adjust_sharpness, (0.3,)),
        (F_t.autocontrast, ()),
        (F_t.equalize, ()),
    ],
)
def test_assert_image_tensor(device, func, args):
    shape = (100,)
    tensor = torch.rand(*shape, dtype=torch.float, device=device)
    with pytest.raises(Exception, match=r"Tensor is not a torch image."):
        func(tensor, *args)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_vflip(device):
    script_vflip = torch.jit.script(F.vflip)

    img_tensor, pil_img = _create_data(16, 18, device=device)
    vflipped_img = F.vflip(img_tensor)
    vflipped_pil_img = F.vflip(pil_img)
    _assert_equal_tensor_to_pil(vflipped_img, vflipped_pil_img)

    # scriptable function test
    vflipped_img_script = script_vflip(img_tensor)
    assert_equal(vflipped_img, vflipped_img_script)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    _test_fn_on_batch(batch_tensors, F.vflip)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_hflip(device):
    script_hflip = torch.jit.script(F.hflip)

    img_tensor, pil_img = _create_data(16, 18, device=device)
    hflipped_img = F.hflip(img_tensor)
    hflipped_pil_img = F.hflip(pil_img)
    _assert_equal_tensor_to_pil(hflipped_img, hflipped_pil_img)

    # scriptable function test
    hflipped_img_script = script_hflip(img_tensor)
    assert_equal(hflipped_img, hflipped_img_script)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    _test_fn_on_batch(batch_tensors, F.hflip)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize(
    "top, left, height, width",
    [
        (1, 2, 4, 5),  # crop inside top-left corner
        (2, 12, 3, 4),  # crop inside top-right corner
        (8, 3, 5, 6),  # crop inside bottom-left corner
        (8, 11, 4, 3),  # crop inside bottom-right corner
        (50, 50, 10, 10),  # crop outside the image
        (-50, -50, 10, 10),  # crop outside the image
    ],
)
def test_crop(device, top, left, height, width):
    script_crop = torch.jit.script(F.crop)

    img_tensor, pil_img = _create_data(16, 18, device=device)

    pil_img_cropped = F.crop(pil_img, top, left, height, width)

    img_tensor_cropped = F.crop(img_tensor, top, left, height, width)
    _assert_equal_tensor_to_pil(img_tensor_cropped, pil_img_cropped)

    img_tensor_cropped = script_crop(img_tensor, top, left, height, width)
    _assert_equal_tensor_to_pil(img_tensor_cropped, pil_img_cropped)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    _test_fn_on_batch(batch_tensors, F.crop, top=top, left=left, height=height, width=width)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("image_size", ("small", "large"))
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("ksize", [(3, 3), [3, 5], (23, 23)])
@pytest.mark.parametrize("sigma", [[0.5, 0.5], (0.5, 0.5), (0.8, 0.8), (1.7, 1.7)])
@pytest.mark.parametrize("fn", [F.gaussian_blur, torch.jit.script(F.gaussian_blur)])
def test_gaussian_blur(device, image_size, dt, ksize, sigma, fn):

    # true_cv2_results = {
    #     # np_img = np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.8)
    #     "3_3_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 3), sigmaX=0.5)
    #     "3_3_0.5": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.8)
    #     "3_5_0.8": ...
    #     # cv2.GaussianBlur(np_img, ksize=(3, 5), sigmaX=0.5)
    #     "3_5_0.5": ...
    #     # np_img2 = np.arange(26 * 28, dtype="uint8").reshape((26, 28))
    #     # cv2.GaussianBlur(np_img2, ksize=(23, 23), sigmaX=1.7)
    #     "23_23_1.7": ...
    # }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gaussian_blur_opencv_results.pt")
    true_cv2_results = torch.load(p)

    if image_size == "small":
        tensor = (
            torch.from_numpy(np.arange(3 * 10 * 12, dtype="uint8").reshape((10, 12, 3))).permute(2, 0, 1).to(device)
        )
    else:
        tensor = torch.from_numpy(np.arange(26 * 28, dtype="uint8").reshape((1, 26, 28))).to(device)

    if dt == torch.float16 and device == "cpu":
        # skip float16 on CPU case
        return

    if dt is not None:
        tensor = tensor.to(dtype=dt)

    _ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
    _sigma = sigma[0] if sigma is not None else None
    shape = tensor.shape
    gt_key = f"{shape[-2]}_{shape[-1]}_{shape[-3]}__{_ksize[0]}_{_ksize[1]}_{_sigma}"
    if gt_key not in true_cv2_results:
        return

    true_out = (
        torch.tensor(true_cv2_results[gt_key]).reshape(shape[-2], shape[-1], shape[-3]).permute(2, 0, 1).to(tensor)
    )

    out = fn(tensor, kernel_size=ksize, sigma=sigma)
    torch.testing.assert_close(out, true_out, rtol=0.0, atol=1.0, msg=f"{ksize}, {sigma}")


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_hsv2rgb(device):
    scripted_fn = torch.jit.script(F_t._hsv2rgb)
    shape = (3, 100, 150)
    for _ in range(10):
        hsv_img = torch.rand(*shape, dtype=torch.float, device=device)
        rgb_img = F_t._hsv2rgb(hsv_img)
        ft_img = rgb_img.permute(1, 2, 0).flatten(0, 1)

        (
            h,
            s,
            v,
        ) = hsv_img.unbind(0)
        h = h.flatten().cpu().numpy()
        s = s.flatten().cpu().numpy()
        v = v.flatten().cpu().numpy()

        rgb = []
        for h1, s1, v1 in zip(h, s, v):
            rgb.append(colorsys.hsv_to_rgb(h1, s1, v1))
        colorsys_img = torch.tensor(rgb, dtype=torch.float32, device=device)
        torch.testing.assert_close(ft_img, colorsys_img, rtol=0.0, atol=1e-5)

        s_rgb_img = scripted_fn(hsv_img)
        torch.testing.assert_close(rgb_img, s_rgb_img)

    batch_tensors = _create_data_batch(120, 100, num_samples=4, device=device).float()
    _test_fn_on_batch(batch_tensors, F_t._hsv2rgb)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_rgb2hsv(device):
    scripted_fn = torch.jit.script(F_t._rgb2hsv)
    shape = (3, 150, 100)
    for _ in range(10):
        rgb_img = torch.rand(*shape, dtype=torch.float, device=device)
        hsv_img = F_t._rgb2hsv(rgb_img)
        ft_hsv_img = hsv_img.permute(1, 2, 0).flatten(0, 1)

        (
            r,
            g,
            b,
        ) = rgb_img.unbind(dim=-3)
        r = r.flatten().cpu().numpy()
        g = g.flatten().cpu().numpy()
        b = b.flatten().cpu().numpy()

        hsv = []
        for r1, g1, b1 in zip(r, g, b):
            hsv.append(colorsys.rgb_to_hsv(r1, g1, b1))

        colorsys_img = torch.tensor(hsv, dtype=torch.float32, device=device)

        ft_hsv_img_h, ft_hsv_img_sv = torch.split(ft_hsv_img, [1, 2], dim=1)
        colorsys_img_h, colorsys_img_sv = torch.split(colorsys_img, [1, 2], dim=1)

        max_diff_h = ((colorsys_img_h * 2 * math.pi).sin() - (ft_hsv_img_h * 2 * math.pi).sin()).abs().max()
        max_diff_sv = (colorsys_img_sv - ft_hsv_img_sv).abs().max()
        max_diff = max(max_diff_h, max_diff_sv)
        assert max_diff < 1e-5

        s_hsv_img = scripted_fn(rgb_img)
        torch.testing.assert_close(hsv_img, s_hsv_img, rtol=1e-5, atol=1e-7)

    batch_tensors = _create_data_batch(120, 100, num_samples=4, device=device).float()
    _test_fn_on_batch(batch_tensors, F_t._rgb2hsv)


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("num_output_channels", (3, 1))
def test_rgb_to_grayscale(device, num_output_channels):
    script_rgb_to_grayscale = torch.jit.script(F.rgb_to_grayscale)

    img_tensor, pil_img = _create_data(32, 34, device=device)

    gray_pil_image = F.rgb_to_grayscale(pil_img, num_output_channels=num_output_channels)
    gray_tensor = F.rgb_to_grayscale(img_tensor, num_output_channels=num_output_channels)

    _assert_approx_equal_tensor_to_pil(gray_tensor.float(), gray_pil_image, tol=1.0 + 1e-10, agg_method="max")

    s_gray_tensor = script_rgb_to_grayscale(img_tensor, num_output_channels=num_output_channels)
    assert_equal(s_gray_tensor, gray_tensor)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    _test_fn_on_batch(batch_tensors, F.rgb_to_grayscale, num_output_channels=num_output_channels)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_center_crop(device):
    script_center_crop = torch.jit.script(F.center_crop)

    img_tensor, pil_img = _create_data(32, 34, device=device)

    cropped_pil_image = F.center_crop(pil_img, [10, 11])

    cropped_tensor = F.center_crop(img_tensor, [10, 11])
    _assert_equal_tensor_to_pil(cropped_tensor, cropped_pil_image)

    cropped_tensor = script_center_crop(img_tensor, [10, 11])
    _assert_equal_tensor_to_pil(cropped_tensor, cropped_pil_image)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    _test_fn_on_batch(batch_tensors, F.center_crop, output_size=[10, 11])


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_five_crop(device):
    script_five_crop = torch.jit.script(F.five_crop)

    img_tensor, pil_img = _create_data(32, 34, device=device)

    cropped_pil_images = F.five_crop(pil_img, [10, 11])

    cropped_tensors = F.five_crop(img_tensor, [10, 11])
    for i in range(5):
        _assert_equal_tensor_to_pil(cropped_tensors[i], cropped_pil_images[i])

    cropped_tensors = script_five_crop(img_tensor, [10, 11])
    for i in range(5):
        _assert_equal_tensor_to_pil(cropped_tensors[i], cropped_pil_images[i])

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    tuple_transformed_batches = F.five_crop(batch_tensors, [10, 11])
    for i in range(len(batch_tensors)):
        img_tensor = batch_tensors[i, ...]
        tuple_transformed_imgs = F.five_crop(img_tensor, [10, 11])
        assert len(tuple_transformed_imgs) == len(tuple_transformed_batches)

        for j in range(len(tuple_transformed_imgs)):
            true_transformed_img = tuple_transformed_imgs[j]
            transformed_img = tuple_transformed_batches[j][i, ...]
            assert_equal(true_transformed_img, transformed_img)

    # scriptable function test
    s_tuple_transformed_batches = script_five_crop(batch_tensors, [10, 11])
    for transformed_batch, s_transformed_batch in zip(tuple_transformed_batches, s_tuple_transformed_batches):
        assert_equal(transformed_batch, s_transformed_batch)


@pytest.mark.parametrize("device", cpu_and_cuda())
def test_ten_crop(device):
    script_ten_crop = torch.jit.script(F.ten_crop)

    img_tensor, pil_img = _create_data(32, 34, device=device)

    cropped_pil_images = F.ten_crop(pil_img, [10, 11])

    cropped_tensors = F.ten_crop(img_tensor, [10, 11])
    for i in range(10):
        _assert_equal_tensor_to_pil(cropped_tensors[i], cropped_pil_images[i])

    cropped_tensors = script_ten_crop(img_tensor, [10, 11])
    for i in range(10):
        _assert_equal_tensor_to_pil(cropped_tensors[i], cropped_pil_images[i])

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    tuple_transformed_batches = F.ten_crop(batch_tensors, [10, 11])
    for i in range(len(batch_tensors)):
        img_tensor = batch_tensors[i, ...]
        tuple_transformed_imgs = F.ten_crop(img_tensor, [10, 11])
        assert len(tuple_transformed_imgs) == len(tuple_transformed_batches)

        for j in range(len(tuple_transformed_imgs)):
            true_transformed_img = tuple_transformed_imgs[j]
            transformed_img = tuple_transformed_batches[j][i, ...]
            assert_equal(true_transformed_img, transformed_img)

    # scriptable function test
    s_tuple_transformed_batches = script_ten_crop(batch_tensors, [10, 11])
    for transformed_batch, s_transformed_batch in zip(tuple_transformed_batches, s_tuple_transformed_batches):
        assert_equal(transformed_batch, s_transformed_batch)


def test_elastic_transform_asserts():
    with pytest.raises(TypeError, match="Argument displacement should be a Tensor"):
        _ = F.elastic_transform("abc", displacement=None)

    with pytest.raises(TypeError, match="img should be PIL Image or Tensor"):
        _ = F.elastic_transform("abc", displacement=torch.rand(1))

    img_tensor = torch.rand(1, 3, 32, 24)
    with pytest.raises(ValueError, match="Argument displacement shape should"):
        _ = F.elastic_transform(img_tensor, displacement=torch.rand(1, 2))


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("interpolation", [NEAREST, BILINEAR, BICUBIC])
@pytest.mark.parametrize("dt", [None, torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize(
    "fill",
    [None, [255, 255, 255], (2.0,)],
)
def test_elastic_transform_consistency(device, interpolation, dt, fill):
    script_elastic_transform = torch.jit.script(F.elastic_transform)
    img_tensor, _ = _create_data(32, 34, device=device)
    # As there is no PIL implementation for elastic_transform,
    # thus we do not run tests tensor vs pillow

    if dt is not None:
        img_tensor = img_tensor.to(dt)

    displacement = T.ElasticTransform.get_params([1.5, 1.5], [2.0, 2.0], [32, 34])
    kwargs = dict(
        displacement=displacement,
        interpolation=interpolation,
        fill=fill,
    )

    out_tensor1 = F.elastic_transform(img_tensor, **kwargs)
    out_tensor2 = script_elastic_transform(img_tensor, **kwargs)
    assert_equal(out_tensor1, out_tensor2)

    batch_tensors = _create_data_batch(16, 18, num_samples=4, device=device)
    displacement = T.ElasticTransform.get_params([1.5, 1.5], [2.0, 2.0], [16, 18])
    kwargs["displacement"] = displacement
    if dt is not None:
        batch_tensors = batch_tensors.to(dt)
    _test_fn_on_batch(batch_tensors, F.elastic_transform, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
