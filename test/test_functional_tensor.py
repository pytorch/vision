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

    def _test_vflip(self, device):
        script_vflip = torch.jit.script(F_t.vflip)
        img_tensor = torch.randn(3, 16, 16, device=device)
        img_tensor_clone = img_tensor.clone()
        vflipped_img = F_t.vflip(img_tensor)
        vflipped_img_again = F_t.vflip(vflipped_img)
        self.assertEqual(vflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, vflipped_img_again))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        vflipped_img_script = script_vflip(img_tensor)
        self.assertTrue(torch.equal(vflipped_img, vflipped_img_script))

    def test_vflip_cpu(self):
        self._test_vflip("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_vflip_cuda(self):
        self._test_vflip("cuda")

    def _test_hflip(self, device):
        script_hflip = torch.jit.script(F_t.hflip)
        img_tensor = torch.randn(3, 16, 16, device=device)
        img_tensor_clone = img_tensor.clone()
        hflipped_img = F_t.hflip(img_tensor)
        hflipped_img_again = F_t.hflip(hflipped_img)
        self.assertEqual(hflipped_img.shape, img_tensor.shape)
        self.assertTrue(torch.equal(img_tensor, hflipped_img_again))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        hflipped_img_script = script_hflip(img_tensor)
        self.assertTrue(torch.equal(hflipped_img, hflipped_img_script))

    def test_hflip_cpu(self):
        self._test_hflip("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_hflip_cuda(self):
        self._test_hflip("cuda")

    def _test_crop(self, device):
        script_crop = torch.jit.script(F.crop)

        img_tensor, pil_img = self._create_data(16, 18, device=device)

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

    def test_crop_cpu(self):
        self._test_crop("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_crop_cuda(self):
        self._test_crop("cuda")

    def test_hsv2rgb(self):
        shape = (3, 100, 150)
        for _ in range(20):
            img = torch.rand(*shape, dtype=torch.float)
            ft_img = F_t._hsv2rgb(img).permute(1, 2, 0).flatten(0, 1)

            h, s, v, = img.unbind(0)
            h = h.flatten().numpy()
            s = s.flatten().numpy()
            v = v.flatten().numpy()

            rgb = []
            for h1, s1, v1 in zip(h, s, v):
                rgb.append(colorsys.hsv_to_rgb(h1, s1, v1))

            colorsys_img = torch.tensor(rgb, dtype=torch.float32)
            max_diff = (ft_img - colorsys_img).abs().max()
            self.assertLess(max_diff, 1e-5)

    def test_rgb2hsv(self):
        shape = (3, 150, 100)
        for _ in range(20):
            img = torch.rand(*shape, dtype=torch.float)
            ft_hsv_img = F_t._rgb2hsv(img).permute(1, 2, 0).flatten(0, 1)

            r, g, b, = img.unbind(0)
            r = r.flatten().numpy()
            g = g.flatten().numpy()
            b = b.flatten().numpy()

            hsv = []
            for r1, g1, b1 in zip(r, g, b):
                hsv.append(colorsys.rgb_to_hsv(r1, g1, b1))

            colorsys_img = torch.tensor(hsv, dtype=torch.float32)

            ft_hsv_img_h, ft_hsv_img_sv = torch.split(ft_hsv_img, [1, 2], dim=1)
            colorsys_img_h, colorsys_img_sv = torch.split(colorsys_img, [1, 2], dim=1)

            max_diff_h = ((colorsys_img_h * 2 * math.pi).sin() - (ft_hsv_img_h * 2 * math.pi).sin()).abs().max()
            max_diff_sv = (colorsys_img_sv - ft_hsv_img_sv).abs().max()
            max_diff = max(max_diff_h, max_diff_sv)

            self.assertLess(max_diff, 1e-5)

    def _test_adjustments(self, device):
        script_adjust_brightness = torch.jit.script(F_t.adjust_brightness)
        script_adjust_contrast = torch.jit.script(F_t.adjust_contrast)
        script_adjust_saturation = torch.jit.script(F_t.adjust_saturation)

        fns = ((F.adjust_brightness, F_t.adjust_brightness, script_adjust_brightness),
               (F.adjust_contrast, F_t.adjust_contrast, script_adjust_contrast),
               (F.adjust_saturation, F_t.adjust_saturation, script_adjust_saturation))

        for _ in range(20):
            channels = 3
            dims = torch.randint(1, 50, (2,))
            shape = (channels, dims[0], dims[1])

            if torch.randint(0, 2, (1,)) == 0:
                img = torch.rand(*shape, dtype=torch.float, device=device)
            else:
                img = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

            factor = 3 * torch.rand(1).item()
            img_clone = img.clone()
            for f, ft, sft in fns:

                ft_img = ft(img, factor).cpu()
                sft_img = sft(img, factor).cpu()
                if not img.dtype.is_floating_point:
                    ft_img = ft_img.to(torch.float) / 255
                    sft_img = sft_img.to(torch.float) / 255

                img_pil = transforms.ToPILImage()(img)
                f_img_pil = f(img_pil, factor)
                f_img = transforms.ToTensor()(f_img_pil)

                # F uses uint8 and F_t uses float, so there is a small
                # difference in values caused by (at most 5) truncations.
                max_diff = (ft_img - f_img).abs().max()
                max_diff_scripted = (sft_img - f_img).abs().max()
                self.assertLess(max_diff, 5 / 255 + 1e-5)
                self.assertLess(max_diff_scripted, 5 / 255 + 1e-5)
                self.assertTrue(torch.equal(img, img_clone))

            # test for class interface
            f = transforms.ColorJitter(brightness=factor)
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

            f = transforms.ColorJitter(contrast=factor)
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

            f = transforms.ColorJitter(saturation=factor)
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

        f = transforms.ColorJitter(brightness=1)
        scripted_fn = torch.jit.script(f)
        scripted_fn(img)

    def test_adjustments(self):
        self._test_adjustments("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_adjustments_cuda(self):
        self._test_adjustments("cuda")

    def test_rgb_to_grayscale(self):
        script_rgb_to_grayscale = torch.jit.script(F_t.rgb_to_grayscale)
        img_tensor = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
        img_tensor_clone = img_tensor.clone()
        grayscale_tensor = F_t.rgb_to_grayscale(img_tensor).to(int)
        grayscale_pil_img = torch.tensor(np.array(F.to_grayscale(F.to_pil_image(img_tensor)))).to(int)
        max_diff = (grayscale_tensor - grayscale_pil_img).abs().max()
        self.assertLess(max_diff, 1.0001)
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        grayscale_script = script_rgb_to_grayscale(img_tensor).to(int)
        self.assertTrue(torch.equal(grayscale_script, grayscale_tensor))

    def _test_center_crop(self, device):
        script_center_crop = torch.jit.script(F.center_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=device)

        cropped_pil_image = F.center_crop(pil_img, [10, 11])

        cropped_tensor = F.center_crop(img_tensor, [10, 11])
        self.compareTensorToPIL(cropped_tensor, cropped_pil_image)

        cropped_tensor = script_center_crop(img_tensor, [10, 11])
        self.compareTensorToPIL(cropped_tensor, cropped_pil_image)

    def test_center_crop(self):
        self._test_center_crop("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_center_crop_cuda(self):
        self._test_center_crop("cuda")

    def _test_five_crop(self, device):
        script_five_crop = torch.jit.script(F.five_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=device)

        cropped_pil_images = F.five_crop(pil_img, [10, 11])

        cropped_tensors = F.five_crop(img_tensor, [10, 11])
        for i in range(5):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

        cropped_tensors = script_five_crop(img_tensor, [10, 11])
        for i in range(5):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

    def test_five_crop(self):
        self._test_five_crop("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_five_crop_cuda(self):
        self._test_five_crop("cuda")

    def _test_ten_crop(self, device):
        script_ten_crop = torch.jit.script(F.ten_crop)

        img_tensor, pil_img = self._create_data(32, 34, device=device)

        cropped_pil_images = F.ten_crop(pil_img, [10, 11])

        cropped_tensors = F.ten_crop(img_tensor, [10, 11])
        for i in range(10):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

        cropped_tensors = script_ten_crop(img_tensor, [10, 11])
        for i in range(10):
            self.compareTensorToPIL(cropped_tensors[i], cropped_pil_images[i])

    def test_ten_crop(self):
        self._test_ten_crop("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_ten_crop_cuda(self):
        self._test_ten_crop("cuda")

    def _test_pad(self, device):
        script_fn = torch.jit.script(F_t.pad)
        tensor, pil_img = self._create_data(7, 8, device=device)

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

    def test_pad_cpu(self):
        self._test_pad("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_pad_cuda(self):
        self._test_pad("cuda")

    def _test_adjust_gamma(self, device):
        script_fn = torch.jit.script(F.adjust_gamma)
        tensor, pil_img = self._create_data(26, 36, device=device)

        for dt in [torch.float64, torch.float32, None]:

            if dt is not None:
                tensor = F.convert_image_dtype(tensor, dt)

            gammas = [0.8, 1.0, 1.2]
            gains = [0.7, 1.0, 1.3]
            for gamma, gain in zip(gammas, gains):

                adjusted_tensor = F.adjust_gamma(tensor, gamma, gain)
                adjusted_pil = F.adjust_gamma(pil_img, gamma, gain)
                scripted_result = script_fn(tensor, gamma, gain)
                self.assertEqual(adjusted_tensor.dtype, scripted_result.dtype)
                self.assertEqual(adjusted_tensor.size()[1:], adjusted_pil.size[::-1])

                rbg_tensor = adjusted_tensor
                if adjusted_tensor.dtype != torch.uint8:
                    rbg_tensor = F.convert_image_dtype(adjusted_tensor, torch.uint8)

                self.compareTensorToPIL(rbg_tensor, adjusted_pil)

                self.assertTrue(adjusted_tensor.allclose(scripted_result))

    def test_adjust_gamma_cpu(self):
        self._test_adjust_gamma("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_adjust_gamma_cuda(self):
        self._test_adjust_gamma("cuda")

    def _test_resize(self, device):
        script_fn = torch.jit.script(F_t.resize)
        tensor, pil_img = self._create_data(26, 36, device=device)

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

    def test_resize_cpu(self):
        self._test_resize("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_resize_cuda(self):
        self._test_resize("cuda")

    def _test_resized_crop(self, device):
        # test values of F.resized_crop in several cases:
        # 1) resize to the same size, crop to the same size => should be identity
        tensor, _ = self._create_data(26, 36, device=device)
        for i in [0, 2, 3]:
            out_tensor = F.resized_crop(tensor, top=0, left=0, height=26, width=36, size=[26, 36], interpolation=i)
            self.assertTrue(tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5]))

        # 2) resize by half and crop a TL corner
        tensor, _ = self._create_data(26, 36, device=device)
        out_tensor = F.resized_crop(tensor, top=0, left=0, height=20, width=30, size=[10, 15], interpolation=0)
        expected_out_tensor = tensor[:, :20:2, :30:2]
        self.assertTrue(
            expected_out_tensor.equal(out_tensor),
            msg="{} vs {}".format(expected_out_tensor[0, :10, :10], out_tensor[0, :10, :10])
        )

    def test_resized_crop_cpu(self):
        self._test_resized_crop("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_resized_crop_cuda(self):
        self._test_resized_crop("cuda")

    def _test_affine(self, device):
        # Tests on square and rectangular images
        scripted_affine = torch.jit.script(F.affine)

        for tensor, pil_img in [self._create_data(26, 26, device=device), self._create_data(32, 26, device=device)]:

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
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1))).to(device)

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
                        tol = 0.06 if device == "cuda" else 0.05
                        self.assertLess(
                            ratio_diff_pixels,
                            tol,
                            msg="{}: {}\n{} vs \n{}".format(
                                (r, a, t, s, sh), ratio_diff_pixels, out_tensor[0, :7, :7], out_pil_tensor[0, :7, :7]
                            )
                        )

    def test_affine_cpu(self):
        self._test_affine("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_affine_cuda(self):
        self._test_affine("cuda")

    def _test_rotate(self, device):
        # Tests on square image
        scripted_rotate = torch.jit.script(F.rotate)

        for tensor, pil_img in [self._create_data(26, 26, device=device), self._create_data(32, 26, device=device)]:

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

    def test_rotate_cpu(self):
        self._test_rotate("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_rotate_cuda(self):
        self._test_rotate("cuda")

    def _test_perspective(self, device):

        from torchvision.transforms import RandomPerspective

        for tensor, pil_img in [self._create_data(26, 34, device=device), self._create_data(26, 26, device=device)]:

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

    def test_perspective_cpu(self):
        self._test_perspective("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
    def test_perspective_cuda(self):
        self._test_perspective("cuda")


if __name__ == '__main__':
    unittest.main()
