import os
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

import numpy as np

import unittest
from typing import Sequence

from common_utils import (
    get_tmp_dir,
    int_dtypes,
    float_dtypes,
    _create_data,
    _create_data_batch,
    _assert_equal_tensor_to_pil,
    _assert_approx_equal_tensor_to_pil,
)
from _assert_utils import assert_equal


NEAREST, BILINEAR, BICUBIC = InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC


def _test_transform_vs_scripted(transform, s_transform, tensor, msg=None):
    torch.manual_seed(12)
    out1 = transform(tensor)
    torch.manual_seed(12)
    out2 = s_transform(tensor)
    assert_equal(out1, out2, msg=msg)


def _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors, msg=None):
    torch.manual_seed(12)
    transformed_batch = transform(batch_tensors)

    for i in range(len(batch_tensors)):
        img_tensor = batch_tensors[i, ...]
        torch.manual_seed(12)
        transformed_img = transform(img_tensor)
        assert_equal(transformed_img, transformed_batch[i, ...], msg=msg)

    torch.manual_seed(12)
    s_transformed_batch = s_transform(batch_tensors)
    assert_equal(transformed_batch, s_transformed_batch, msg=msg)


def _test_functional_op(f, device, fn_kwargs=None, test_exact_match=True, **match_kwargs):
    fn_kwargs = fn_kwargs or {}

    tensor, pil_img = _create_data(height=10, width=10, device=device)
    transformed_tensor = f(tensor, **fn_kwargs)
    transformed_pil_img = f(pil_img, **fn_kwargs)
    if test_exact_match:
        _assert_equal_tensor_to_pil(transformed_tensor, transformed_pil_img, **match_kwargs)
    else:
        _assert_approx_equal_tensor_to_pil(transformed_tensor, transformed_pil_img, **match_kwargs)


def _test_class_op(method, device, meth_kwargs=None, test_exact_match=True, **match_kwargs):
    #TODO: change the name: it's not a method, it's a class.
    meth_kwargs = meth_kwargs or {}

    # test for class interface
    f = method(**meth_kwargs)
    scripted_fn = torch.jit.script(f)

    tensor, pil_img = _create_data(26, 34, device=device)
    # set seed to reproduce the same transformation for tensor and PIL image
    torch.manual_seed(12)
    transformed_tensor = f(tensor)
    torch.manual_seed(12)
    transformed_pil_img = f(pil_img)
    if test_exact_match:
        _assert_equal_tensor_to_pil(transformed_tensor, transformed_pil_img, **match_kwargs)
    else:
        _assert_approx_equal_tensor_to_pil(transformed_tensor.float(), transformed_pil_img, **match_kwargs)

    torch.manual_seed(12)
    transformed_tensor_script = scripted_fn(tensor)
    assert_equal(transformed_tensor, transformed_tensor_script)

    batch_tensors = _create_data_batch(height=23, width=34, channels=3, num_samples=4, device=device)
    _test_transform_vs_scripted_on_batch(f, scripted_fn, batch_tensors)

    with get_tmp_dir() as tmp_dir:
        scripted_fn.save(os.path.join(tmp_dir, f"t_{method.__name__}.pt"))


def _test_op(func, method, device, fn_kwargs=None, meth_kwargs=None, test_exact_match=True, **match_kwargs):
    _test_functional_op(func, device, fn_kwargs, test_exact_match=test_exact_match, **match_kwargs)
    _test_class_op(method, device, meth_kwargs, test_exact_match=test_exact_match, **match_kwargs)


class Tester(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"

    def test_random_horizontal_flip(self):
        _test_op(F.hflip, T.RandomHorizontalFlip, device=self.device)

    def test_random_vertical_flip(self):
        _test_op(F.vflip, T.RandomVerticalFlip, device=self.device)

    def test_random_invert(self):
        _test_op(F.invert, T.RandomInvert, device=self.device)

    def test_random_posterize(self):
        fn_kwargs = meth_kwargs = {"bits": 4}
        _test_op(
            F.posterize, T.RandomPosterize, device=self.device, fn_kwargs=fn_kwargs,
            meth_kwargs=meth_kwargs
        )

    def test_random_solarize(self):
        fn_kwargs = meth_kwargs = {"threshold": 192.0}
        _test_op(
            F.solarize, T.RandomSolarize, device=self.device, fn_kwargs=fn_kwargs,
            meth_kwargs=meth_kwargs
        )

    def test_random_adjust_sharpness(self):
        fn_kwargs = meth_kwargs = {"sharpness_factor": 2.0}
        _test_op(
            F.adjust_sharpness, T.RandomAdjustSharpness, device=self.device, fn_kwargs=fn_kwargs,
            meth_kwargs=meth_kwargs
        )

    def test_random_autocontrast(self):
        # We check the max abs difference because on some (very rare) pixels, the actual value may be different
        # between PIL and tensors due to floating approximations.
        _test_op(
            F.autocontrast, T.RandomAutocontrast, device=self.device, test_exact_match=False,
            agg_method='max', tol=(1 + 1e-5), allowed_percentage_diff=.05
        )

    def test_random_equalize(self):
        _test_op(F.equalize, T.RandomEqualize, device=self.device)

    def test_color_jitter(self):

        tol = 1.0 + 1e-10
        for f in [0.1, 0.5, 1.0, 1.34, (0.3, 0.7), [0.4, 0.5]]:
            meth_kwargs = {"brightness": f}
            _test_class_op(
                T.ColorJitter, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
                tol=tol, agg_method="max"
            )

        for f in [0.2, 0.5, 1.0, 1.5, (0.3, 0.7), [0.4, 0.5]]:
            meth_kwargs = {"contrast": f}
            _test_class_op(
                T.ColorJitter, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
                tol=tol, agg_method="max"
            )

        for f in [0.5, 0.75, 1.0, 1.25, (0.3, 0.7), [0.3, 0.4]]:
            meth_kwargs = {"saturation": f}
            _test_class_op(
                T.ColorJitter, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
                tol=tol, agg_method="max"
            )

        for f in [0.2, 0.5, (-0.2, 0.3), [-0.4, 0.5]]:
            meth_kwargs = {"hue": f}
            _test_class_op(
                T.ColorJitter, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
                tol=16.1, agg_method="max"
            )

        # All 4 parameters together
        meth_kwargs = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.2}
        _test_class_op(
            T.ColorJitter, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
            tol=12.1, agg_method="max"
        )

    def test_pad(self):
        for m in ["constant", "edge", "reflect", "symmetric"]:
            fill = 127 if m == "constant" else 0
            for mul in [1, -1]:
                # Test functional.pad (PIL and Tensor) with padding as single int
                _test_functional_op(
                    F.pad, fn_kwargs={"padding": mul * 2, "fill": fill, "padding_mode": m},
                    device=self.device
                )
                # Test functional.pad and transforms.Pad with padding as [int, ]
                fn_kwargs = meth_kwargs = {"padding": [mul * 2, ], "fill": fill, "padding_mode": m}
                _test_op(
                    F.pad, T.Pad, device=self.device, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
                )
                # Test functional.pad and transforms.Pad with padding as list
                fn_kwargs = meth_kwargs = {"padding": [mul * 4, 4], "fill": fill, "padding_mode": m}
                _test_op(
                    F.pad, T.Pad, device=self.device, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
                )
                # Test functional.pad and transforms.Pad with padding as tuple
                fn_kwargs = meth_kwargs = {"padding": (mul * 2, 2, 2, mul * 2), "fill": fill, "padding_mode": m}
                _test_op(
                    F.pad, T.Pad, device=self.device, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
                )

    def test_crop(self):
        fn_kwargs = {"top": 2, "left": 3, "height": 4, "width": 5}
        # Test transforms.RandomCrop with size and padding as tuple
        meth_kwargs = {"size": (4, 5), "padding": (4, 4), "pad_if_needed": True, }
        _test_op(
            F.crop, T.RandomCrop, device=self.device, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )

        # Test transforms.functional.crop including outside the image area
        fn_kwargs = {"top": -2, "left": 3, "height": 4, "width": 5}  # top
        _test_functional_op(F.crop, fn_kwargs=fn_kwargs, device=self.device)

        fn_kwargs = {"top": 1, "left": -3, "height": 4, "width": 5}  # left
        _test_functional_op(F.crop, fn_kwargs=fn_kwargs, device=self.device)

        fn_kwargs = {"top": 7, "left": 3, "height": 4, "width": 5}  # bottom
        _test_functional_op(F.crop, fn_kwargs=fn_kwargs, device=self.device)

        fn_kwargs = {"top": 3, "left": 8, "height": 4, "width": 5}  # right
        _test_functional_op(F.crop, fn_kwargs=fn_kwargs, device=self.device)

        fn_kwargs = {"top": -3, "left": -3, "height": 15, "width": 15}  # all
        _test_functional_op(F.crop, fn_kwargs=fn_kwargs, device=self.device)

        sizes = [5, [5, ], [6, 6]]
        padding_configs = [
            {"padding_mode": "constant", "fill": 0},
            {"padding_mode": "constant", "fill": 10},
            {"padding_mode": "constant", "fill": 20},
            {"padding_mode": "edge"},
            {"padding_mode": "reflect"},
        ]

        for size in sizes:
            for padding_config in padding_configs:
                config = dict(padding_config)
                config["size"] = size
                _test_class_op(T.RandomCrop, self.device, config)

    def test_center_crop(self):
        fn_kwargs = {"output_size": (4, 5)}
        meth_kwargs = {"size": (4, 5), }
        _test_op(
            F.center_crop, T.CenterCrop, device=self.device, fn_kwargs=fn_kwargs,
            meth_kwargs=meth_kwargs
        )
        fn_kwargs = {"output_size": (5,)}
        meth_kwargs = {"size": (5, )}
        _test_op(
            F.center_crop, T.CenterCrop, device=self.device, fn_kwargs=fn_kwargs,
            meth_kwargs=meth_kwargs
        )
        tensor = torch.randint(0, 256, (3, 10, 10), dtype=torch.uint8, device=self.device)
        # Test torchscript of transforms.CenterCrop with size as int
        f = T.CenterCrop(size=5)
        scripted_fn = torch.jit.script(f)
        scripted_fn(tensor)

        # Test torchscript of transforms.CenterCrop with size as [int, ]
        f = T.CenterCrop(size=[5, ])
        scripted_fn = torch.jit.script(f)
        scripted_fn(tensor)

        # Test torchscript of transforms.CenterCrop with size as tuple
        f = T.CenterCrop(size=(6, 6))
        scripted_fn = torch.jit.script(f)
        scripted_fn(tensor)

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_center_crop.pt"))

    def _test_op_list_output(self, func, method, out_length, fn_kwargs=None, meth_kwargs=None):
        if fn_kwargs is None:
            fn_kwargs = {}
        if meth_kwargs is None:
            meth_kwargs = {}

        fn = getattr(F, func)
        scripted_fn = torch.jit.script(fn)

        tensor, pil_img = _create_data(height=20, width=20, device=self.device)
        transformed_t_list = fn(tensor, **fn_kwargs)
        transformed_p_list = fn(pil_img, **fn_kwargs)
        self.assertEqual(len(transformed_t_list), len(transformed_p_list))
        self.assertEqual(len(transformed_t_list), out_length)
        for transformed_tensor, transformed_pil_img in zip(transformed_t_list, transformed_p_list):
            _assert_equal_tensor_to_pil(transformed_tensor, transformed_pil_img)

        transformed_t_list_script = scripted_fn(tensor.detach().clone(), **fn_kwargs)
        self.assertEqual(len(transformed_t_list), len(transformed_t_list_script))
        self.assertEqual(len(transformed_t_list_script), out_length)
        for transformed_tensor, transformed_tensor_script in zip(transformed_t_list, transformed_t_list_script):
            assert_equal(
                transformed_tensor,
                transformed_tensor_script,
                msg="{} vs {}".format(transformed_tensor, transformed_tensor_script),
            )

        # test for class interface
        fn = getattr(T, method)(**meth_kwargs)
        scripted_fn = torch.jit.script(fn)
        output = scripted_fn(tensor)
        self.assertEqual(len(output), len(transformed_t_list_script))

        # test on batch of tensors
        batch_tensors = _create_data_batch(height=23, width=34, channels=3, num_samples=4, device=self.device)
        torch.manual_seed(12)
        transformed_batch_list = fn(batch_tensors)

        for i in range(len(batch_tensors)):
            img_tensor = batch_tensors[i, ...]
            torch.manual_seed(12)
            transformed_img_list = fn(img_tensor)
            for transformed_img, transformed_batch in zip(transformed_img_list, transformed_batch_list):
                assert_equal(
                    transformed_img,
                    transformed_batch[i, ...],
                    msg="{} vs {}".format(transformed_img, transformed_batch[i, ...]),
                )

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_op_list_{}.pt".format(method)))

    def test_five_crop(self):
        fn_kwargs = meth_kwargs = {"size": (5,)}
        self._test_op_list_output(
            "five_crop", "FiveCrop", out_length=5, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": [5, ]}
        self._test_op_list_output(
            "five_crop", "FiveCrop", out_length=5, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": (4, 5)}
        self._test_op_list_output(
            "five_crop", "FiveCrop", out_length=5, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": [4, 5]}
        self._test_op_list_output(
            "five_crop", "FiveCrop", out_length=5, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )

    def test_ten_crop(self):
        fn_kwargs = meth_kwargs = {"size": (5,)}
        self._test_op_list_output(
            "ten_crop", "TenCrop", out_length=10, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": [5, ]}
        self._test_op_list_output(
            "ten_crop", "TenCrop", out_length=10, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": (4, 5)}
        self._test_op_list_output(
            "ten_crop", "TenCrop", out_length=10, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = meth_kwargs = {"size": [4, 5]}
        self._test_op_list_output(
            "ten_crop", "TenCrop", out_length=10, fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )

    def test_resize(self):

        # TODO: Minimal check for bug-fix, improve this later
        x = torch.rand(3, 32, 46)
        t = T.Resize(size=38)
        y = t(x)
        # If size is an int, smaller edge of the image will be matched to this number.
        # i.e, if height > width, then image will be rescaled to (size * height / width, size).
        self.assertTrue(isinstance(y, torch.Tensor))
        self.assertEqual(y.shape[1], 38)
        self.assertEqual(y.shape[2], int(38 * 46 / 32))

        tensor, _ = _create_data(height=34, width=36, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        for dt in [None, torch.float32, torch.float64]:
            if dt is not None:
                # This is a trivial cast to float of uint8 data to test all cases
                tensor = tensor.to(dt)
            for size in [32, 34, [32, ], [32, 32], (32, 32), [34, 35]]:
                for max_size in (None, 35, 1000):
                    if max_size is not None and isinstance(size, Sequence) and len(size) != 1:
                        continue  # Not supported
                    for interpolation in [BILINEAR, BICUBIC, NEAREST]:

                        if isinstance(size, int):
                            script_size = [size, ]
                        else:
                            script_size = size

                        transform = T.Resize(size=script_size, interpolation=interpolation, max_size=max_size)
                        s_transform = torch.jit.script(transform)
                        _test_transform_vs_scripted(transform, s_transform, tensor)
                        _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            s_transform.save(os.path.join(tmp_dir, "t_resize.pt"))

    def test_resized_crop(self):
        tensor = torch.randint(0, 256, size=(3, 44, 56), dtype=torch.uint8, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        for scale in [(0.7, 1.2), [0.7, 1.2]]:
            for ratio in [(0.75, 1.333), [0.75, 1.333]]:
                for size in [(32, ), [44, ], [32, ], [32, 32], (32, 32), [44, 55]]:
                    for interpolation in [NEAREST, BILINEAR, BICUBIC]:
                        transform = T.RandomResizedCrop(
                            size=size, scale=scale, ratio=ratio, interpolation=interpolation
                        )
                        s_transform = torch.jit.script(transform)
                        _test_transform_vs_scripted(transform, s_transform, tensor)
                        _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            s_transform.save(os.path.join(tmp_dir, "t_resized_crop.pt"))

    def test_random_affine(self):
        tensor = torch.randint(0, 256, size=(3, 44, 56), dtype=torch.uint8, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        def _test(**kwargs):
            transform = T.RandomAffine(**kwargs)
            s_transform = torch.jit.script(transform)

            _test_transform_vs_scripted(transform, s_transform, tensor)
            _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

            return s_transform

        for interpolation in [NEAREST, BILINEAR]:
            for shear in [15, 10.0, (5.0, 10.0), [-15, 15], [-10.0, 10.0, -11.0, 11.0]]:
                _test(degrees=0.0, interpolation=interpolation, shear=shear)

            for scale in [(0.7, 1.2), [0.7, 1.2]]:
                _test(degrees=0.0, interpolation=interpolation, scale=scale)

            for translate in [(0.1, 0.2), [0.2, 0.1]]:
                _test(degrees=0.0, interpolation=interpolation, translate=translate)

            for degrees in [45, 35.0, (-45, 45), [-90.0, 90.0]]:
                _test(degrees=degrees, interpolation=interpolation)

            for fill in [85, (10, -10, 10), 0.7, [0.0, 0.0, 0.0], [1, ], 1]:
                _test(degrees=0.0, interpolation=interpolation, fill=fill)

        s_transform = _test(degrees=0.0)
        with get_tmp_dir() as tmp_dir:
            s_transform.save(os.path.join(tmp_dir, "t_random_affine.pt"))

    def test_random_rotate(self):
        tensor = torch.randint(0, 256, size=(3, 44, 56), dtype=torch.uint8, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        for center in [(0, 0), [10, 10], None, (56, 44)]:
            for expand in [True, False]:
                for degrees in [45, 35.0, (-45, 45), [-90.0, 90.0]]:
                    for interpolation in [NEAREST, BILINEAR]:
                        for fill in [85, (10, -10, 10), 0.7, [0.0, 0.0, 0.0], [1, ], 1]:
                            transform = T.RandomRotation(
                                degrees=degrees, interpolation=interpolation, expand=expand, center=center, fill=fill
                            )
                            s_transform = torch.jit.script(transform)

                            _test_transform_vs_scripted(transform, s_transform, tensor)
                            _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            s_transform.save(os.path.join(tmp_dir, "t_random_rotate.pt"))

    def test_random_perspective(self):
        tensor = torch.randint(0, 256, size=(3, 44, 56), dtype=torch.uint8, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        for distortion_scale in np.linspace(0.1, 1.0, num=20):
            for interpolation in [NEAREST, BILINEAR]:
                for fill in [85, (10, -10, 10), 0.7, [0.0, 0.0, 0.0], [1, ], 1]:
                    transform = T.RandomPerspective(
                        distortion_scale=distortion_scale,
                        interpolation=interpolation,
                        fill=fill
                    )
                    s_transform = torch.jit.script(transform)

                    _test_transform_vs_scripted(transform, s_transform, tensor)
                    _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            s_transform.save(os.path.join(tmp_dir, "t_perspective.pt"))

    def test_to_grayscale(self):

        meth_kwargs = {"num_output_channels": 1}
        tol = 1.0 + 1e-10
        _test_class_op(
            T.Grayscale, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
            tol=tol, agg_method="max"
        )

        meth_kwargs = {"num_output_channels": 3}
        _test_class_op(
            T.Grayscale, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
            tol=tol, agg_method="max"
        )

        meth_kwargs = {}
        _test_class_op(
            T.RandomGrayscale, meth_kwargs=meth_kwargs, test_exact_match=False, device=self.device,
            tol=tol, agg_method="max"
        )

    def test_normalize(self):
        fn = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor, _ = _create_data(26, 34, device=self.device)

        with self.assertRaisesRegex(TypeError, r"Input tensor should be a float tensor"):
            fn(tensor)

        batch_tensors = torch.rand(4, 3, 44, 56, device=self.device)
        tensor = tensor.to(dtype=torch.float32) / 255.0
        # test for class interface
        scripted_fn = torch.jit.script(fn)

        _test_transform_vs_scripted(fn, scripted_fn, tensor)
        _test_transform_vs_scripted_on_batch(fn, scripted_fn, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_norm.pt"))

    def test_linear_transformation(self):
        c, h, w = 3, 24, 32

        tensor, _ = _create_data(h, w, channels=c, device=self.device)

        matrix = torch.rand(c * h * w, c * h * w, device=self.device)
        mean_vector = torch.rand(c * h * w, device=self.device)

        fn = T.LinearTransformation(matrix, mean_vector)
        scripted_fn = torch.jit.script(fn)

        _test_transform_vs_scripted(fn, scripted_fn, tensor)

        batch_tensors = torch.rand(4, c, h, w, device=self.device)
        # We skip some tests from _test_transform_vs_scripted_on_batch as
        # results for scripted and non-scripted transformations are not exactly the same
        torch.manual_seed(12)
        transformed_batch = fn(batch_tensors)
        torch.manual_seed(12)
        s_transformed_batch = scripted_fn(batch_tensors)
        assert_equal(transformed_batch, s_transformed_batch)

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_norm.pt"))

    def test_compose(self):
        tensor, _ = _create_data(26, 34, device=self.device)
        tensor = tensor.to(dtype=torch.float32) / 255.0

        transforms = T.Compose([
            T.CenterCrop(10),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        s_transforms = torch.nn.Sequential(*transforms.transforms)

        scripted_fn = torch.jit.script(s_transforms)
        torch.manual_seed(12)
        transformed_tensor = transforms(tensor)
        torch.manual_seed(12)
        transformed_tensor_script = scripted_fn(tensor)
        assert_equal(transformed_tensor, transformed_tensor_script, msg="{}".format(transforms))

        t = T.Compose([
            lambda x: x,
        ])
        with self.assertRaisesRegex(RuntimeError, r"Could not get name of python class object"):
            torch.jit.script(t)

    def test_random_apply(self):
        tensor, _ = _create_data(26, 34, device=self.device)
        tensor = tensor.to(dtype=torch.float32) / 255.0

        transforms = T.RandomApply([
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
        ], p=0.4)
        s_transforms = T.RandomApply(torch.nn.ModuleList([
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
        ]), p=0.4)

        scripted_fn = torch.jit.script(s_transforms)
        torch.manual_seed(12)
        transformed_tensor = transforms(tensor)
        torch.manual_seed(12)
        transformed_tensor_script = scripted_fn(tensor)
        assert_equal(transformed_tensor, transformed_tensor_script, msg="{}".format(transforms))

        if torch.device(self.device).type == "cpu":
            # Can't check this twice, otherwise
            # "Can't redefine method: forward on class: __torch__.torchvision.transforms.transforms.RandomApply"
            transforms = T.RandomApply([
                T.ColorJitter(),
            ], p=0.3)
            with self.assertRaisesRegex(RuntimeError, r"Module 'RandomApply' has no attribute 'transforms'"):
                torch.jit.script(transforms)

    def test_gaussian_blur(self):
        tol = 1.0 + 1e-10
        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": 3, "sigma": 0.75},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": 23, "sigma": [0.1, 2.0]},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": 23, "sigma": (0.1, 2.0)},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": [3, 3], "sigma": (1.0, 1.0)},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": (3, 3), "sigma": (0.1, 2.0)},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

        _test_class_op(
            T.GaussianBlur, meth_kwargs={"kernel_size": [23], "sigma": 0.75},
            test_exact_match=False, device=self.device, agg_method="max", tol=tol
        )

    def test_random_erasing(self):
        img = torch.rand(3, 60, 60)

        # Test Set 0: invalid value
        random_erasing = T.RandomErasing(value=(0.1, 0.2, 0.3, 0.4), p=1.0)
        with self.assertRaises(ValueError, msg="If value is a sequence, it should have either a single value or 3"):
            random_erasing(img)

        tensor, _ = _create_data(24, 32, channels=3, device=self.device)
        batch_tensors = torch.rand(4, 3, 44, 56, device=self.device)

        test_configs = [
            {"value": 0.2},
            {"value": "random"},
            {"value": (0.2, 0.2, 0.2)},
            {"value": "random", "ratio": (0.1, 0.2)},
        ]

        for config in test_configs:
            fn = T.RandomErasing(**config)
            scripted_fn = torch.jit.script(fn)
            _test_transform_vs_scripted(fn, scripted_fn, tensor)
            _test_transform_vs_scripted_on_batch(fn, scripted_fn, batch_tensors)

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_random_erasing.pt"))

    def test_convert_image_dtype(self):
        tensor, _ = _create_data(26, 34, device=self.device)
        batch_tensors = torch.rand(4, 3, 44, 56, device=self.device)

        for in_dtype in int_dtypes() + float_dtypes():
            in_tensor = tensor.to(in_dtype)
            in_batch_tensors = batch_tensors.to(in_dtype)
            for out_dtype in int_dtypes() + float_dtypes():

                fn = T.ConvertImageDtype(dtype=out_dtype)
                scripted_fn = torch.jit.script(fn)

                if (in_dtype == torch.float32 and out_dtype in (torch.int32, torch.int64)) or \
                        (in_dtype == torch.float64 and out_dtype == torch.int64):
                    with self.assertRaisesRegex(RuntimeError, r"cannot be performed safely"):
                        _test_transform_vs_scripted(fn, scripted_fn, in_tensor)
                    with self.assertRaisesRegex(RuntimeError, r"cannot be performed safely"):
                        _test_transform_vs_scripted_on_batch(fn, scripted_fn, in_batch_tensors)
                    continue

                _test_transform_vs_scripted(fn, scripted_fn, in_tensor)
                _test_transform_vs_scripted_on_batch(fn, scripted_fn, in_batch_tensors)

        with get_tmp_dir() as tmp_dir:
            scripted_fn.save(os.path.join(tmp_dir, "t_convert_dtype.pt"))

    def test_autoaugment(self):
        tensor = torch.randint(0, 256, size=(3, 44, 56), dtype=torch.uint8, device=self.device)
        batch_tensors = torch.randint(0, 256, size=(4, 3, 44, 56), dtype=torch.uint8, device=self.device)

        s_transform = None
        for policy in T.AutoAugmentPolicy:
            for fill in [None, 85, (10, -10, 10), 0.7, [0.0, 0.0, 0.0], [1, ], 1]:
                transform = T.AutoAugment(policy=policy, fill=fill)
                s_transform = torch.jit.script(transform)
                for _ in range(25):
                    _test_transform_vs_scripted(transform, s_transform, tensor)
                    _test_transform_vs_scripted_on_batch(transform, s_transform, batch_tensors)

        if s_transform is not None:
            with get_tmp_dir() as tmp_dir:
                s_transform.save(os.path.join(tmp_dir, "t_autoaugment.pt"))


@unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
class CUDATester(Tester):

    def setUp(self):
        torch.set_deterministic(False)
        self.device = "cuda"


if __name__ == '__main__':
    unittest.main()
