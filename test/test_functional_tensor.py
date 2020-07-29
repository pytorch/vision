import unittest
import random
import colorsys
import math

from PIL import Image
from PIL.Image import NEAREST, BILINEAR, BICUBIC

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional as F


class Tester(unittest.TestCase):

    def _create_data(self, height=3, width=3, channels=3):
        tensor = torch.randint(0, 255, (channels, height, width), dtype=torch.uint8)
        pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())
        return tensor, pil_img

    def compareTensorToPIL(self, tensor, pil_image, msg=None):
        pil_tensor = torch.as_tensor(np.array(pil_image).transpose((2, 0, 1)))
        if msg is None:
            msg = "tensor:\n{} \ndid not equal PIL tensor:\n{}".format(tensor, pil_tensor)
        self.assertTrue(tensor.equal(pil_tensor), msg)

    def approxEqualTensorToPIL(self, tensor, pil_image, tol=1e-5, msg=None):
        pil_tensor = torch.as_tensor(np.array(pil_image).transpose((2, 0, 1))).to(tensor)
        mae = torch.abs(tensor - pil_tensor).mean().item()
        self.assertTrue(
            mae < tol,
            msg="{}: mae={}, tol={}: \n{}\nvs\n{}".format(msg, mae, tol, tensor[0, :10, :10], pil_tensor[0, :10, :10])
        )

    def test_vflip(self):
        script_vflip = torch.jit.script(F_t.vflip)
        img_tensor = torch.randn(3, 16, 16)
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
        img_tensor = torch.randn(3, 16, 16)
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
        script_crop = torch.jit.script(F_t.crop)
        img_tensor = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
        img_tensor_clone = img_tensor.clone()
        top = random.randint(0, 15)
        left = random.randint(0, 15)
        height = random.randint(1, 16 - top)
        width = random.randint(1, 16 - left)
        img_cropped = F_t.crop(img_tensor, top, left, height, width)
        img_PIL = transforms.ToPILImage()(img_tensor)
        img_PIL_cropped = F.crop(img_PIL, top, left, height, width)
        img_cropped_GT = transforms.ToTensor()(img_PIL_cropped)
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        self.assertTrue(torch.equal(img_cropped, (img_cropped_GT * 255).to(torch.uint8)),
                        "functional_tensor crop not working")
        # scriptable function test
        cropped_img_script = script_crop(img_tensor, top, left, height, width)
        self.assertTrue(torch.equal(img_cropped, cropped_img_script))

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

    def test_adjustments(self):
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
                img = torch.rand(*shape, dtype=torch.float)
            else:
                img = torch.randint(0, 256, shape, dtype=torch.uint8)

            factor = 3 * torch.rand(1)
            img_clone = img.clone()
            for f, ft, sft in fns:

                ft_img = ft(img, factor)
                sft_img = sft(img, factor)
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
            f = transforms.ColorJitter(brightness=factor.item())
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

            f = transforms.ColorJitter(contrast=factor.item())
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

            f = transforms.ColorJitter(saturation=factor.item())
            scripted_fn = torch.jit.script(f)
            scripted_fn(img)

        f = transforms.ColorJitter(brightness=1)
        scripted_fn = torch.jit.script(f)
        scripted_fn(img)

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

    def test_center_crop(self):
        script_center_crop = torch.jit.script(F_t.center_crop)
        img_tensor = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)
        img_tensor_clone = img_tensor.clone()
        cropped_tensor = F_t.center_crop(img_tensor, [10, 10])
        cropped_pil_image = F.center_crop(transforms.ToPILImage()(img_tensor), [10, 10])
        cropped_pil_tensor = (transforms.ToTensor()(cropped_pil_image) * 255).to(torch.uint8)
        self.assertTrue(torch.equal(cropped_tensor, cropped_pil_tensor))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        cropped_script = script_center_crop(img_tensor, [10, 10])
        self.assertTrue(torch.equal(cropped_script, cropped_tensor))

    def test_five_crop(self):
        script_five_crop = torch.jit.script(F_t.five_crop)
        img_tensor = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)
        img_tensor_clone = img_tensor.clone()
        cropped_tensor = F_t.five_crop(img_tensor, [10, 10])
        cropped_pil_image = F.five_crop(transforms.ToPILImage()(img_tensor), [10, 10])
        self.assertTrue(torch.equal(cropped_tensor[0],
                                    (transforms.ToTensor()(cropped_pil_image[0]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[1],
                                    (transforms.ToTensor()(cropped_pil_image[2]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[2],
                                    (transforms.ToTensor()(cropped_pil_image[1]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[3],
                                    (transforms.ToTensor()(cropped_pil_image[3]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[4],
                                    (transforms.ToTensor()(cropped_pil_image[4]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        cropped_script = script_five_crop(img_tensor, [10, 10])
        for cropped_script_img, cropped_tensor_img in zip(cropped_script, cropped_tensor):
            self.assertTrue(torch.equal(cropped_script_img, cropped_tensor_img))

    def test_ten_crop(self):
        script_ten_crop = torch.jit.script(F_t.ten_crop)
        img_tensor = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)
        img_tensor_clone = img_tensor.clone()
        cropped_tensor = F_t.ten_crop(img_tensor, [10, 10])
        cropped_pil_image = F.ten_crop(transforms.ToPILImage()(img_tensor), [10, 10])
        self.assertTrue(torch.equal(cropped_tensor[0],
                                    (transforms.ToTensor()(cropped_pil_image[0]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[1],
                                    (transforms.ToTensor()(cropped_pil_image[2]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[2],
                                    (transforms.ToTensor()(cropped_pil_image[1]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[3],
                                    (transforms.ToTensor()(cropped_pil_image[3]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[4],
                                    (transforms.ToTensor()(cropped_pil_image[4]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[5],
                                    (transforms.ToTensor()(cropped_pil_image[5]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[6],
                                    (transforms.ToTensor()(cropped_pil_image[7]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[7],
                                    (transforms.ToTensor()(cropped_pil_image[6]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[8],
                                    (transforms.ToTensor()(cropped_pil_image[8]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(cropped_tensor[9],
                                    (transforms.ToTensor()(cropped_pil_image[9]) * 255).to(torch.uint8)))
        self.assertTrue(torch.equal(img_tensor, img_tensor_clone))
        # scriptable function test
        cropped_script = script_ten_crop(img_tensor, [10, 10])
        for cropped_script_img, cropped_tensor_img in zip(cropped_script, cropped_tensor):
            self.assertTrue(torch.equal(cropped_script_img, cropped_tensor_img))

    def test_pad(self):
        script_fn = torch.jit.script(F_t.pad)
        tensor, pil_img = self._create_data(7, 8)

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

    def test_adjust_gamma(self):
        script_fn = torch.jit.script(F_t.adjust_gamma)
        tensor, pil_img = self._create_data(26, 36)

        for dt in [torch.float64, torch.float32, None]:

            if dt is not None:
                tensor = F.convert_image_dtype(tensor, dt)

            gammas = [0.8, 1.0, 1.2]
            gains = [0.7, 1.0, 1.3]
            for gamma, gain in zip(gammas, gains):

                adjusted_tensor = F_t.adjust_gamma(tensor, gamma, gain)
                adjusted_pil = F_pil.adjust_gamma(pil_img, gamma, gain)
                scripted_result = script_fn(tensor, gamma, gain)
                self.assertEqual(adjusted_tensor.dtype, scripted_result.dtype)
                self.assertEqual(adjusted_tensor.size()[1:], adjusted_pil.size[::-1])

                rbg_tensor = adjusted_tensor
                if adjusted_tensor.dtype != torch.uint8:
                    rbg_tensor = F.convert_image_dtype(adjusted_tensor, torch.uint8)

                self.compareTensorToPIL(rbg_tensor, adjusted_pil)

                self.assertTrue(adjusted_tensor.equal(scripted_result))

    def test_resize(self):
        script_fn = torch.jit.script(F_t.resize)
        tensor, pil_img = self._create_data(26, 36)

        for dt in [None, torch.float32, torch.float64]:
            if dt is not None:
                # This is a trivial cast to float of uint8 data to test all cases
                tensor = tensor.to(dt)
            for size in [32, [32, ], [32, 32], (32, 32), ]:
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
        tensor, _ = self._create_data(26, 36)
        for i in [0, 2, 3]:
            out_tensor = F.resized_crop(tensor, top=0, left=0, height=26, width=36, size=[26, 36], interpolation=i)
            self.assertTrue(tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5]))

        # 2) resize by half and crop a TL corner
        tensor, _ = self._create_data(26, 36)
        out_tensor = F.resized_crop(tensor, top=0, left=0, height=20, width=30, size=[10, 15], interpolation=0)
        expected_out_tensor = tensor[:, :20:2, :30:2]
        self.assertTrue(
            expected_out_tensor.equal(out_tensor),
            msg="{} vs {}".format(expected_out_tensor[0, :10, :10], out_tensor[0, :10, :10])
        )

    def test_affine(self):
        # Tests on square image
        tensor, pil_img = self._create_data(26, 26)

        scripted_affine = torch.jit.script(F.affine)
        # 1) identity map
        out_tensor = F.affine(tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
        self.assertTrue(
            tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5])
        )
        out_tensor = scripted_affine(tensor, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
        self.assertTrue(
            tensor.equal(out_tensor), msg="{} vs {}".format(out_tensor[0, :5, :5], tensor[0, :5, :5])
        )

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
            for fn in [F.affine, scripted_affine]:
                out_tensor = fn(tensor, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
                if true_tensor is not None:
                    self.assertTrue(
                        true_tensor.equal(out_tensor),
                        msg="{}\n{} vs \n{}".format(a, out_tensor[0, :5, :5], true_tensor[0, :5, :5])
                    )
                else:
                    true_tensor = out_tensor

                out_pil_img = F.affine(pil_img, angle=a, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], resample=0)
                out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

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
        # 3) Test translation
        test_configs = [
            [10, 12], (12, 13)
        ]
        for t in test_configs:
            for fn in [F.affine, scripted_affine]:
                out_tensor = fn(tensor, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], resample=0)
                out_pil_img = F.affine(pil_img, angle=0, translate=t, scale=1.0, shear=[0.0, 0.0], resample=0)
                self.compareTensorToPIL(out_tensor, out_pil_img)

        # 3) Test rotation + translation + scale + share
        test_configs = [
            (45, [5, 6], 1.0, [0.0, 0.0]),
            (33, (5, -4), 1.0, [0.0, 0.0]),
            (45, [5, 4], 1.2, [0.0, 0.0]),
            (33, (4, 8), 2.0, [0.0, 0.0]),
            (85, (10, -10), 0.7, [0.0, 0.0]),
            (0, [0, 0], 1.0, [35.0, ]),
            (25, [0, 0], 1.2, [0.0, 15.0]),
            (45, [10, 0], 0.7, [2.0, 5.0]),
            (45, [10, -10], 1.2, [4.0, 5.0]),
        ]
        for r in [0, ]:
            for a, t, s, sh in test_configs:
                for fn in [F.affine, scripted_affine]:
                    out_tensor = fn(tensor, angle=a, translate=t, scale=s, shear=sh, resample=r)
                    out_pil_img = F.affine(pil_img, angle=a, translate=t, scale=s, shear=sh, resample=r)
                    out_pil_tensor = torch.from_numpy(np.array(out_pil_img).transpose((2, 0, 1)))

                    num_diff_pixels = (out_tensor != out_pil_tensor).sum().item() / 3.0
                    ratio_diff_pixels = num_diff_pixels / out_tensor.shape[-1] / out_tensor.shape[-2]
                    # Tolerance : less than 5% of different pixels
                    self.assertLess(
                        ratio_diff_pixels,
                        0.05,
                        msg="{}: {}\n{} vs \n{}".format(
                            (r, a, t, s, sh), ratio_diff_pixels, out_tensor[0, :7, :7], out_pil_tensor[0, :7, :7]
                        )
                    )


if __name__ == '__main__':
    unittest.main()
