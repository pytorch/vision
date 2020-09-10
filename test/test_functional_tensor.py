import unittest
import colorsys
import math

import numpy as np
from PIL.Image import NEAREST, BILINEAR, BICUBIC

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional as F

from common_utils import TransformsTester


class Tester(TransformsTester):

    def setUp(self):
        self.device = "cpu"

    def test_vflip(self):
        script_vflip = torch.jit.script(F_t.vflip)
        img_tensor = torch.randn(3, 16, 16, device=self.device)
        img_tensor_clone = img_tensor.clone()
        vflipped_img = F_t.vflip(img_tensor)
        vflipped_img_again = F_t.vflip(vflipped_img)
        self.assertEqual(vflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, vflipped_img_again))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        vflipped_img_script = script_vflip(img_tensor)
        self.assertTrue(torch.equal(vflipped_img, vflipped_img_script))

    def test_hflip(self):
        script_hflip = torch.jit.script(F_t.hflip)
        img_tensor = torch.randn(3, 16, 16, device=self.device)
        img_tensor_clone = img_tensor.clone()
        hflipped_img = F_t.hflip(img_tensor)
        hflipped_img_again = F_t.hflip(hflipped_img)
        self.assertEqual(hflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, hflipped_img_again))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        hflipped_img_script = script_hflip(img_tensor)
        self.assertTrue(torch.equal(hflipped_img, hflipped_img_script))

    def test_crop(self):
        script_crop = torch.jit.script(F.crop)

        img_tensor, pil_img = self._create_data(16, 18, device=self.device)

        test_configs = [
            (1, 2, 4, 5),   # crop inside top-left corner
            (2, 12, 3, 4),  # crop inside top-right corner
            (8, 3, 5, 6),   # crop inside bottom-left corner
            (8, 11, 4, 3),  # crop inside bottom-right corner
        ]

        for top, left, height, width in test_configs:
            pil_img_cropped = F.crop(pil_img, top, left, height, width)

            img_tensor_cropped = F.crop(img_tensor, top, left, height, width)
            self.compareTensorToPIL(img_tensor_cropped, pil_img_cropped)

            img_tensor_cropped = script_crop(img_tensor, top, left, height, width)
            self.compareTensorToPIL(img_tensor_cropped, pil_img_cropped)

    def test_hsv2rgb(self):
        scripted_fn = torch.jit.script(F_t._hsv2rgb)
        shape = (3, 100, 150)
        for _ in range(10):
            hsv_img = torch.rand(*shape, dtype=torch.float, device=self.device)
            rgb_img = F_t._hsv2rgb(hsv_img)
            ft_img = rgb_img.permute(1, 2, 0).flatten(0, 1)

            h, s, v, = hsv_img.unbind(0)
            h = h.flatten().cpu().numpy()
            s = s.flatten().cpu().numpy()
            v = v.flatten().cpu().numpy()

            rgb = []
            for h1, s1, v1 in zip(h, s, v):
                rgb.append(colorsys.hsv_to_rgb(h1, s1, v1))
            colorsys_img = torch.tensor(rgb, dtype=torch.float32, device=self.device)
            max_diff = (ft_img - colorsys_img).abs().max()
            self.assertLess(max_diff, 1e-5)

            s_rgb_img = scripted_fn(hsv_img)
            self.assertTrue(rgb_img.allclose(s_rgb_img))

    def test_rgb2hsv(self):
        scripted_fn = torch.jit.script(F_t._rgb2hsv)
        shape = (3, 150, 100)
        for _ in range(10):
            rgb_img = torch.rand(*shape, dtype=torch.float, device=self.device)
            hsv_img = F_t._rgb2hsv(rgb_img)
            ft_hsv_img = hsv_img.permute(1, 2, 0).flatten(0, 1)

            r, g, b, = rgb_img.unbind(0)
            r = r.flatten().cpu().numpy()
            g = g.flatten().cpu().numpy()
            b = b.flatten().cpu().numpy()

            hsv = []
            for r1, g1, b1 in zip(r, g, b):
                hsv.append(colorsys.rgb_to_hsv(r1, g1, b1))

            colorsys_img = torch.tensor(hsv, dtype=torch.float32, device=self.device)

            ft_hsv_img_h, ft_hsv_img_sv = torch.split(ft_hsv_img, [1, 2], dim=1)
            colorsys_img_h, colorsys_img_sv = torch.split(colorsys_img, [1, 2], dim=1)

            max_diff_h = ((colorsys_img_h * 2 * math.pi).sin() - (ft_hsv_img_h * 2 * math.pi).sin()).abs().max()
            max_diff_sv = (colorsys_img_sv - ft_hsv_img_sv).abs().max()
            max_diff = max(max_diff_h, max_diff_sv)
            self.assertLess(max_diff, 1e-5)

            s_hsv_img = scripted_fn(rgb_img)
            self.assertTrue(hsv_img.allclose(s_hsv_img))

    def test_rgb_to_grayscale(self):
        script_rgb_to_grayscale = torch.jit.script(F.rgb_to_grayscale)

        img_tensor, pil_img = self._create_data(32, 34, device=self.device)

        for num_output_channels in (3, 1):
            gray_pil_image = F.rgb_to_grayscale(pil_img, num_output_channels=num_output_channels)
            gray_tensor = F.rgb_to_grayscale(img_tensor, num_output_channels=num_output_channels)

            if num_output_channels == 1:
                print(gray_tensor.shape)

            self.approxEqualTensorToPIL(gray_tensor.float(), gray_pil_image, tol=1.0 + 1e-10, agg_method="max")

            s_gray_tensor = script_rgb_to_grayscale(img_tensor, num_output_channels=num_output_channels)
            self.assertTrue(s_gray_tensor.equal(gray_tensor))

    def test_center_crop(self):
        script_center_crop = torch.jit.script(F.center_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=self.device)

        cropped_pil_image = F.center_crop(pil_img, [10, 11])

        cropped_tensor = F.center_crop(img_tensor, [10, 11])
        self.compareTensorToPIL(cropped_tensor, cropped_pil_image)

        cropped_tensor = script_center_crop(img_tensor, [10, 11])
        self.compareTensorToPIL(cropped_tensor, cropped_pil_image)

    def test_five_crop(self):
        script_five_crop = torch.jit.script(F.five_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=self.device)

        cropped_pil_images = F.five_crop(pil_img, [10, 11])

        cropped_tensors = F.five_crop(img_tensor, [10, 11])
        for i in range(5):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

        cropped_tensors = script_five_crop(img_tensor, [10, 11])
        for i in range(5):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

    def test_ten_crop(self):
        script_ten_crop = torch.jit.script(F.ten_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=self.device)

        cropped_pil_images = F.ten_crop(pil_img, [10, 11])

        cropped_tensors = F.ten_crop(img_tensor, [10, 11])
        for i in range(10):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

        cropped_tensors = script_ten_crop(img_tensor, [10, 11])
        for i in range(10):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

    def test_pad(self):
        script_fn = torch.jit.script(F_t.pad)
        tensor, pil_img = self._create_data(7, 8, device=self.device)

        for dt in [None, torch.float32, torch.float64]:
            if dt is not None:
                # This is a trivial cast to float of uint8 data to test all cases
                tensor = tensor.to(dt)
            for pad in [2, [3, ], [0, 3], (3, 3), [4, 2, 4, 3]]:
                configs = [
                    {"padding_mode": "constant", "fill": 0},
                    {"padding_mode": "constant", "fill": 10},
                    {"padding_mode": "constant", "fill": 20},
                    {"padding_mode": "edge"},
                    {"padding_mode": "reflect"},
                    {"padding_mode": "symmetric"},
                ]
                for kwargs in configs:
                    pad_tensor = F_t.pad(tensor, pad, **kwargs)
                    pad_pil_img = F_pil.pad(pil_img, pad, **kwargs)

                    pad_tensor_8b = pad_tensor
                    # we need to cast to uint8 to compare with PIL image
                    if pad_tensor_8b.dtype != torch.uint8:
                        pad_tensor_8b = pad_tensor_8b.to(torch.uint8)

                    self.compareTensorToPIL(pad_tensor_8b, pad_pil_img, msg="{}, {}".format(pad, kwargs))

                    if isinstance(pad, int):
                        script_pad = [pad, ]
                    else:
                        script_pad = pad
                    pad_tensor_script = script_fn(tensor, script_pad, **kwargs)
                    self.assertTrue(pad_tensor.equal(pad_tensor_script), msg="{}, {}".format(pad, kwargs))

        with self.assertRaises(ValueError, msg="Padding can not be negative for symmetric padding_mode"):
            F_t.pad(tensor, (-2, -3), padding_mode="symmetric")

    def _test_adjust_fn(self, fn, fn_pil, fn_t, configs, tol=2.0 + 1e-10, agg_method="max"):
        script_fn = torch.jit.script(fn)
        torch.manual_seed(15)
        tensor, pil_img = self._create_data(26, 34, device=self.device)

        for dt in [None, torch.float32, torch.float64]:

            if dt is not None:
                tensor = F.convert_image_dtype(tensor, dt)

            for config in configs:
                adjusted_tensor = fn_t(tensor, **config)
                adjusted_pil = fn_pil(pil_img, **config)
                scripted_result = script_fn(tensor, **config)
                msg = "{}, {}".format(dt, config)
                self.assertEqual(adjusted_tensor.dtype, scripted_result.dtype, msg=msg)
                self.assertEqual(adjusted_tensor.size()[1:], adjusted_pil.size[::-1], msg=msg)

                rbg_tensor = adjusted_tensor

                if adjusted_tensor.dtype != torch.uint8:
                    rbg_tensor = F.convert_image_dtype(adjusted_tensor, torch.uint8)

                # Check that max difference does not exceed 2 in [0, 255] range
                # Exact matching is not possible due to incompatibility convert_image_dtype and PIL results
                self.approxEqualTensorToPIL(rbg_tensor.float(), adjusted_pil, tol=tol, msg=msg, agg_method=agg_method)

                atol = 1e-6
                if adjusted_tensor.dtype == torch.uint8 and "cuda" in torch.device(self.device).type:
                    atol = 1.0
                self.assertTrue(adjusted_tensor.allclose(scripted_result, atol=atol), msg=msg)

    def test_adjust_brightness(self):
        self._test_adjust_fn(
            F.adjust_brightness,
            F_pil.adjust_brightness,
            F_t.adjust_brightness,
            [{"brightness_factor": f} for f in [0.1, 0.5, 1.0, 1.34, 2.5]]
        )

    def test_adjust_contrast(self):
        self._test_adjust_fn(
            F.adjust_contrast,
            F_pil.adjust_contrast,
            F_t.adjust_contrast,
            [{"contrast_factor": f} for f in [0.2, 0.5, 1.0, 1.5, 2.0]]
        )

    def test_adjust_saturation(self):
        self._test_adjust_fn(
            F.adjust_saturation,
            F_pil.adjust_saturation,
            F_t.adjust_saturation,
            [{"saturation_factor": f} for f in [0.5, 0.75, 1.0, 1.5, 2.0]]
        )

    def test_adjust_hue(self):
        self._test_adjust_fn(
            F.adjust_hue,
            F_pil.adjust_hue,
            F_t.adjust_hue,
            [{"hue_factor": f} for f in [-0.45, -0.25, 0.0, 0.25, 0.45]],
            tol=0.1,
            agg_method="mean"
        )

    def test_adjust_gamma(self):
        self._test_adjust_fn(
            F.adjust_gamma,
            F_pil.adjust_gamma,
            F_t.adjust_gamma,
            [{"gamma": g1, "gain": g2} for g1, g2 in zip([0.8, 1.0, 1.2], [0.7, 1.0, 1.3])]
        )

    def test_resize(self):
        script_fn = torch.jit.script(F_t.resize)
        tensor, pil_img = self._create_data(26, 36, device=self.device)

        for dt in [None, torch.float32, torch.float64]:
            if dt is not None:
                # This is a trivial cast to float of uint8 data to test all cases
                tensor = tensor.to(dt)
            for size in [32, 26, [32, ], [32, 32], (32, 32), [26, 35]]:
                for interpolation in [BILINEAR, BICUBIC, NEAREST]:
                    resized_tensor = F_t.resize(tensor, size=size, interpolation=interpolation)
                    resized_pil_img = F_pil.resize(pil_img, size=size, interpolation=interpolation)

                    self.assertEqual(
                        resized_tensor.size()[1:], resized_pil_img.size[::-1], msg="{}, {}".format(size, interpolation)
                    )

                    if interpolation != NEAREST:
                        # We can not check values if mode = NEAREST, as results are different
                        # E.g. resized_tensor  = [[a, a, b, c, d, d, e, ...]]
                        # E.g. resized_pil_img = [[a, b, c, c, d, e, f, ...]]
                        resized_tensor_f = resized_tensor
                        # we need to cast to uint8 to compare with PIL image
                        if resized_tensor_f.dtype == torch.uint8:
                            resized_tensor_f = resized_tensor_f.to(torch.float)

                        # Pay attention to high tolerance for MAE
                        self.approxEqualTensorToPIL(
                            resized_tensor_f, resized_pil_img, tol=8.0, msg="{}, {}".format(size, interpolation)
                        )

                    if isinstance(size, int):
                        script_size = [size, ]
                    else:
                        script_size = size
                    resize_result = script_fn(tensor, size=script_size, interpolation=interpolation)
                    self.assertTrue(resized_tensor.equal(resize_result), msg="{}, {}".format(size, interpolation))

    def test_resized_crop(self):
        # test values of F.resized_crop in several cases:
        # 1) resize to the same size, crop to the same size => should be identity
        tensor, _ = self._create_data(26, 36, device=self.device)
        for i in [0, 2, 3]:
            out_tensor = F.resized_crop(tensor, top=0, left=0, height=26, width=36, size=[26, 36], interpolation=i)
            self.assertTrue(tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5]))

        # 2) resize by half and crop a TL corner
        tensor, _ = self._create_data(26, 36, device=self.device)
        out_tensor = F.resized_crop(tensor, top=0, left=0, height=20, width=30, size=[10, 15], interpolation=0)
        expected_out_tensor = tensor[:, :20:2, :30:2]
        self.assertTrue(
            expected_out_tensor.equal(out_tensor),
            msg="{} vs {}".format(expected_out_tensor[0, :10, :10], out_tensor[0, :10, :10])
        )

    def test_affine(self):
        # Tests on square and rectangular images
        scripted_affine = torch.jit.script(F.affine)

        data = [self._create_data(26, 26, device=self.device), self._create_data(32, 26, device=self.device)]
        for tensor, pil_img in data:

            # 1) identity map
            out_tensor = F.affine(tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
            self.assertTrue(
                tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5])
            )
            out_tensor = scripted_affine(tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
            self.assertTrue(
                tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5])
            )

            if pil_img.size[0] == pil_img.size[1]:
                # 2) Test rotation
                test_configs = [
                    (90, torch.rot90(tensor, k=1, dims=(-1, -2))),
                    (45, None),
                    (30, None),
                    (-30, None),
                    (-45, None),
                    (-90, torch.rot90(tensor, k=-1, dims=(-1, -2))),
                    (180, torch.rot90(tensor, k=2, dims=(-1, -2))),
                ]
                for a, true_tensor in test_configs:

                    out_pil_img = F.affine(
                        pil_img, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0
                    )
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1))).to(self.device)

                    for fn in [F.affine, scripted_affine]:
                        out_tensor = fn(
                            tensor, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0
                        )
                        if true_tensor is not None:
                            self.assertTrue(
                                true_tensor.equal(out_tensor),
                                msg="{}\n{} vs \n{}".format(a, out_tensor[0, :5, :5], true_tensor[0, :5, :5])
                            )
                        else:
                            true_tensor = out_tensor

                        num_diff_pixels = (true_tensor != out_pil_tensor).sum().item() / 3.0
                        ratio_diff_pixels = num_diff_pixels / true_tensor.shape[-1] / true_tensor.shape[-2]
                        # Tolerance : less than 6% of different pixels
                        self.assertLess(
                            ratio_diff_pixels,
                            0.06,
                            msg="{}\n{} vs \n{}".format(
                                ratio_diff_pixels, true_tensor[0, :7, :7], out_pil_tensor[0, :7, :7]
                            )
                        )
            else:
                test_configs = [
                    90, 45, 15, -30, -60, -120
                ]
                for a in test_configs:

                    out_pil_img = F.affine(
                        pil_img, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0
                    )
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

                    for fn in [F.affine, scripted_affine]:
                        out_tensor = fn(
                            tensor, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0
                        ).cpu()

                        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
                        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
                        # Tolerance : less than 3% of different pixels
                        self.assertLess(
                            ratio_diff_pixels,
                            0.03,
                            msg="{}: {}\n{} vs \n{}".format(
                                a, ratio_diff_pixels, out_tensor[0, :7, :7], out_pil_tensor[0, :7, :7]
                            )
                        )

            # 3) Test translation
            test_configs = [
                [10, 12], (-12, -13)
            ]
            for t in test_configs:

                out_pil_img = F.affine(pil_img, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], resample=0)

                for fn in [F.affine, scripted_affine]:
                    out_tensor = fn(tensor, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], resample=0)

                    self.compareTensorToPIL(out_tensor, out_pil_img)

            # 3) Test rotation + translation + scale + share
            test_configs = [
                (45, [5, 6], 1.0, [0.0, 0.0]),
                (33, (5, -4), 1.0, [0.0, 0.0]),
                (45, [-5, 4], 1.2, [0.0, 0.0]),
                (33, (-4, -8), 2.0, [0.0, 0.0]),
                (85, (10, -10), 0.7, [0.0, 0.0]),
                (0, [0, 0], 1.0, [35.0, ]),
                (-25, [0, 0], 1.2, [0.0, 15.0]),
                (-45, [-10, 0], 0.7, [2.0, 5.0]),
                (-45, [-10, -10], 1.2, [4.0, 5.0]),
                (-90, [0, 0], 1.0, [0.0, 0.0]),
            ]
            for r in [0, ]:
                for a, t, s, sh in test_configs:
                    out_pil_img = F.affine(pil_img, angle=a, translate=t, scale=s, shear=sh, resample=r)
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

                    for fn in [F.affine, scripted_affine]:
                        out_tensor = fn(tensor, angle=a, translate=t, scale=s, shear=sh, resample=r).cpu()
                        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
                        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
                        # Tolerance : less than 5% (cpu), 6% (cuda) of different pixels
                        tol = 0.06 if self.device == "cuda" else 0.05
                        self.assertLess(
                            ratio_diff_pixels,
                            tol,
                            msg="{}: {}\n{} vs \n{}".format(
                                (r, a, t, s, sh), ratio_diff_pixels, out_tensor[0, :7, :7], out_pil_tensor[0, :7, :7]
                            )
                        )

    def test_rotate(self):
        # Tests on square image
        scripted_rotate = torch.jit.script(F.rotate)

        data = [self._create_data(26, 26, device=self.device), self._create_data(32, 26, device=self.device)]
        for tensor, pil_img in data:

            img_size = pil_img.size
            centers = [
                None,
                (int(img_size[0] * 0.3), int(img_size[0] * 0.4)),
                [int(img_size[0] * 0.5), int(img_size[0] * 0.6)]
            ]

            for r in [0, ]:
                for a in range(-180, 180, 17):
                    for e in [True, False]:
                        for c in centers:

                            out_pil_img = F.rotate(pil_img, angle=a, resample=r, expand=e, center=c)
                            out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))
                            for fn in [F.rotate, scripted_rotate]:
                                out_tensor = fn(tensor, angle=a, resample=r, expand=e, center=c).cpu()

                                self.assertEqual(
                                    out_tensor.shape,
                                    out_pil_tensor.shape,
                                    msg="{}: {} vs {}".format(
                                        (img_size, r, a, e, c), out_tensor.shape, out_pil_tensor.shape
                                    )
                                )
                                num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
                                ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
                                # Tolerance : less than 2% of different pixels
                                self.assertLess(
                                    ratio_diff_pixels,
                                    0.02,
                                    msg="{}: {}\n{} vs \n{}".format(
                                        (img_size, r, a, e, c),
                                        ratio_diff_pixels,
                                        out_tensor[0, :7, :7],
                                        out_pil_tensor[0, :7, :7]
                                    )
                                )

    def test_perspective(self):

        from torchvision.transforms import RandomPerspective

        data = [self._create_data(26, 34, device=self.device), self._create_data(26, 26, device=self.device)]
        for tensor, pil_img in data:

            scripted_tranform = torch.jit.script(F.perspective)

            test_configs = [
                [[[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]],
                [[[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]],
                [[[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]],
            ]
            n = 10
            test_configs += [
                RandomPerspective.get_params(pil_img.size[0], pil_img.size[1], i / n) for i in range(n)
            ]

            for r in [0, ]:
                for spoints, epoints in test_configs:
                    out_pil_img = F.perspective(pil_img, startpoints=spoints, endpoints=epoints, interpolation=r)
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

                    for fn in [F.perspective, scripted_tranform]:
                        out_tensor = fn(tensor, startpoints=spoints, endpoints=epoints, interpolation=r).cpu()

                        num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
                        ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
                        # Tolerance : less than 5% of different pixels
                        self.assertLess(
                            ratio_diff_pixels,
                            0.05,
                            msg="{}: {}\n{} vs \n{}".format(
                                (r, spoints, epoints),
                                ratio_diff_pixels,
                                out_tensor[0, :7, :7],
                                out_pil_tensor[0, :7, :7]
                            )
                        )


@unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
class CUDATester(Tester):

    def setUp(self):
        self.device = "cuda"


if __name__ == '__main__':
    unittest.main()
