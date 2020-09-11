import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from PIL.Image import NEAREST, BILINEAR, BICUBIC

import numpy as np

import unittest

from common_utils import TransformsTester


class Tester(TransformsTester):

    def setUp(self):
        self.device = "cpu"

    def _test_functional_op(self, func, fn_kwargs):
        if fn_kwargs is None:
            fn_kwargs = {}
        tensor, pil_img = self._create_data(height=10, width=10, device=self.device)
        transformed_tensor = getattr(F, func)(tensor, **fn_kwargs)
        transformed_pil_img = getattr(F, func)(pil_img, **fn_kwargs)
        self.compareTensorToPIL(transformed_tensor, transformed_pil_img)

    def _test_class_op(self, method, meth_kwargs=None, test_exact_match=True, **match_kwargs):
        if meth_kwargs is None:
            meth_kwargs = {}

        tensor, pil_img = self._create_data(26, 34, device=self.device)
        # test for class interface
        f = getattr(T, method)(**meth_kwargs)
        scripted_fn = torch.jit.script(f)

        # set seed to reproduce the same transformation for tensor and PIL image
        torch.manual_seed(12)
        transformed_tensor = f(tensor)
        torch.manual_seed(12)
        transformed_pil_img = f(pil_img)
        if test_exact_match:
            self.compareTensorToPIL(transformed_tensor, transformed_pil_img, **match_kwargs)
        else:
            self.approxEqualTensorToPIL(transformed_tensor.float(), transformed_pil_img, **match_kwargs)

        torch.manual_seed(12)
        transformed_tensor_script = scripted_fn(tensor)
        self.assertTrue(transformed_tensor.equal(transformed_tensor_script))

    def _test_op(self, func, method, fn_kwargs=None, meth_kwargs=None):
        self._test_functional_op(func, fn_kwargs)
        self._test_class_op(method, meth_kwargs)

    def test_random_horizontal_flip(self):
        self._test_op('hflip', 'RandomHorizontalFlip')

    def test_random_vertical_flip(self):
        self._test_op('vflip', 'RandomVerticalFlip')

    def test_color_jitter(self):

        tol = 1.0 + 1e-10
        for f in [0.1, 0.5, 1.0, 1.34, (0.3, 0.7), [0.4, 0.5]]:
            meth_kwargs = {"brightness": f}
            self._test_class_op(
                "ColorJitter", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
            )

        for f in [0.2, 0.5, 1.0, 1.5, (0.3, 0.7), [0.4, 0.5]]:
            meth_kwargs = {"contrast": f}
            self._test_class_op(
                "ColorJitter", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
            )

        for f in [0.5, 0.75, 1.0, 1.25, (0.3, 0.7), [0.3, 0.4]]:
            meth_kwargs = {"saturation": f}
            self._test_class_op(
                "ColorJitter", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
            )

        for f in [0.2, 0.5, (-0.2, 0.3), [-0.4, 0.5]]:
            meth_kwargs = {"hue": f}
            self._test_class_op(
                "ColorJitter", meth_kwargs=meth_kwargs, test_exact_match=False, tol=0.1, agg_method="mean"
            )

        # All 4 parameters together
        meth_kwargs = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.2}
        self._test_class_op(
            "ColorJitter", meth_kwargs=meth_kwargs, test_exact_match=False, tol=0.1, agg_method="mean"
        )

    def test_pad(self):

        # Test functional.pad (PIL and Tensor) with padding as single int
        self._test_functional_op(
            "pad", fn_kwargs={"padding": 2, "fill": 0, "padding_mode": "constant"}
        )
        # Test functional.pad and transforms.Pad with padding as [int, ]
        fn_kwargs = meth_kwargs = {"padding": [2, ], "fill": 0, "padding_mode": "constant"}
        self._test_op(
            "pad", "Pad", fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        # Test functional.pad and transforms.Pad with padding as list
        fn_kwargs = meth_kwargs = {"padding": [4, 4], "fill": 0, "padding_mode": "constant"}
        self._test_op(
            "pad", "Pad", fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        # Test functional.pad and transforms.Pad with padding as tuple
        fn_kwargs = meth_kwargs = {"padding": (2, 2, 2, 2), "fill": 127, "padding_mode": "constant"}
        self._test_op(
            "pad", "Pad", fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )

    def test_crop(self):
        fn_kwargs = {"top": 2, "left": 3, "height": 4, "width": 5}
        # Test transforms.RandomCrop with size and padding as tuple
        meth_kwargs = {"size": (4, 5), "padding": (4, 4), "pad_if_needed": True, }
        self._test_op(
            'crop', 'RandomCrop', fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )

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
                self._test_class_op("RandomCrop", config)

    def test_center_crop(self):
        fn_kwargs = {"output_size": (4, 5)}
        meth_kwargs = {"size": (4, 5), }
        self._test_op(
            "center_crop", "CenterCrop", fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        fn_kwargs = {"output_size": (5,)}
        meth_kwargs = {"size": (5, )}
        self._test_op(
            "center_crop", "CenterCrop", fn_kwargs=fn_kwargs, meth_kwargs=meth_kwargs
        )
        tensor = torch.randint(0, 255, (3, 10, 10), dtype=torch.uint8, device=self.device)
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

    def _test_op_list_output(self, func, method, out_length, fn_kwargs=None, meth_kwargs=None):
        if fn_kwargs is None:
            fn_kwargs = {}
        if meth_kwargs is None:
            meth_kwargs = {}
        tensor, pil_img = self._create_data(height=20, width=20, device=self.device)
        transformed_t_list = getattr(F, func)(tensor, **fn_kwargs)
        transformed_p_list = getattr(F, func)(pil_img, **fn_kwargs)
        self.assertEqual(len(transformed_t_list), len(transformed_p_list))
        self.assertEqual(len(transformed_t_list), out_length)
        for transformed_tensor, transformed_pil_img in zip(transformed_t_list, transformed_p_list):
            self.compareTensorToPIL(transformed_tensor, transformed_pil_img)

        scripted_fn = torch.jit.script(getattr(F, func))
        transformed_t_list_script = scripted_fn(tensor.detach().clone(), **fn_kwargs)
        self.assertEqual(len(transformed_t_list), len(transformed_t_list_script))
        self.assertEqual(len(transformed_t_list_script), out_length)
        for transformed_tensor, transformed_tensor_script in zip(transformed_t_list, transformed_t_list_script):
            self.assertTrue(transformed_tensor.equal(transformed_tensor_script),
                            msg="{} vs {}".format(transformed_tensor, transformed_tensor_script))

        # test for class interface
        f = getattr(T, method)(**meth_kwargs)
        scripted_fn = torch.jit.script(f)
        output = scripted_fn(tensor)
        self.assertEqual(len(output), len(transformed_t_list_script))

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
        tensor, _ = self._create_data(height=34, width=36, device=self.device)
        script_fn = torch.jit.script(F.resize)

        for dt in [None, torch.float32, torch.float64]:
            if dt is not None:
                # This is a trivial cast to float of uint8 data to test all cases
                tensor = tensor.to(dt)
            for size in [32, 34, [32, ], [32, 32], (32, 32), [34, 35]]:
                for interpolation in [BILINEAR, BICUBIC, NEAREST]:

                    resized_tensor = F.resize(tensor, size=size, interpolation=interpolation)

                    if isinstance(size, int):
                        script_size = [size, ]
                    else:
                        script_size = size

                    s_resized_tensor = script_fn(tensor, size=script_size, interpolation=interpolation)
                    self.assertTrue(s_resized_tensor.equal(resized_tensor))

                    transform = T.Resize(size=script_size, interpolation=interpolation)
                    resized_tensor = transform(tensor)
                    script_transform = torch.jit.script(transform)
                    s_resized_tensor = script_transform(tensor)
                    self.assertTrue(s_resized_tensor.equal(resized_tensor))

    def test_resized_crop(self):
        tensor = torch.randint(0, 255, size=(3, 44, 56), dtype=torch.uint8, device=self.device)

        for scale in [(0.7, 1.2), [0.7, 1.2]]:
            for ratio in [(0.75, 1.333), [0.75, 1.333]]:
                for size in [(32, ), [44, ], [32, ], [32, 32], (32, 32), [44, 55]]:
                    for interpolation in [NEAREST, BILINEAR, BICUBIC]:
                        transform = T.RandomResizedCrop(
                            size=size, scale=scale, ratio=ratio, interpolation=interpolation
                        )
                        s_transform = torch.jit.script(transform)

                        torch.manual_seed(12)
                        out1 = transform(tensor)
                        torch.manual_seed(12)
                        out2 = s_transform(tensor)
                        self.assertTrue(out1.equal(out2))

    def test_random_affine(self):
        tensor = torch.randint(0, 255, size=(3, 44, 56), dtype=torch.uint8, device=self.device)

        for shear in [15, 10.0, (5.0, 10.0), [-15, 15], [-10.0, 10.0, -11.0, 11.0]]:
            for scale in [(0.7, 1.2), [0.7, 1.2]]:
                for translate in [(0.1, 0.2), [0.2, 0.1]]:
                    for degrees in [45, 35.0, (-45, 45), [-90.0, 90.0]]:
                        for interpolation in [NEAREST, BILINEAR]:
                            transform = T.RandomAffine(
                                degrees=degrees, translate=translate,
                                scale=scale, shear=shear, resample=interpolation
                            )
                            s_transform = torch.jit.script(transform)

                            torch.manual_seed(12)
                            out1 = transform(tensor)
                            torch.manual_seed(12)
                            out2 = s_transform(tensor)
                            self.assertTrue(out1.equal(out2))

    def test_random_rotate(self):
        tensor = torch.randint(0, 255, size=(3, 44, 56), dtype=torch.uint8, device=self.device)

        for center in [(0, 0), [10, 10], None, (56, 44)]:
            for expand in [True, False]:
                for degrees in [45, 35.0, (-45, 45), [-90.0, 90.0]]:
                    for interpolation in [NEAREST, BILINEAR]:
                        transform = T.RandomRotation(
                            degrees=degrees, resample=interpolation, expand=expand, center=center
                        )
                        s_transform = torch.jit.script(transform)

                        torch.manual_seed(12)
                        out1 = transform(tensor)
                        torch.manual_seed(12)
                        out2 = s_transform(tensor)
                        self.assertTrue(out1.equal(out2))

    def test_random_perspective(self):
        tensor = torch.randint(0, 255, size=(3, 44, 56), dtype=torch.uint8, device=self.device)

        for distortion_scale in np.linspace(0.1, 1.0, num=20):
            for interpolation in [NEAREST, BILINEAR]:
                transform = T.RandomPerspective(
                    distortion_scale=distortion_scale,
                    interpolation=interpolation
                )
                s_transform = torch.jit.script(transform)

                torch.manual_seed(12)
                out1 = transform(tensor)
                torch.manual_seed(12)
                out2 = s_transform(tensor)
                self.assertTrue(out1.equal(out2))

    def test_to_grayscale(self):

        meth_kwargs = {"num_output_channels": 1}
        tol = 1.0 + 1e-10
        self._test_class_op(
            "Grayscale", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
        )

        meth_kwargs = {"num_output_channels": 3}
        self._test_class_op(
            "Grayscale", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
        )

        meth_kwargs = {}
        self._test_class_op(
            "RandomGrayscale", meth_kwargs=meth_kwargs, test_exact_match=False, tol=tol, agg_method="max"
        )


@unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
class CUDATester(Tester):

    def setUp(self):
        self.device = "cuda"


if __name__ == '__main__':
    unittest.main()
