import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional as F
import numpy as np
import unittest
import random
import colorsys

from PIL import Image


class Tester(unittest.TestCase):

    def _create_data(self, height=3, width=3, channels=3):
        tensor = torch.randint(0, 255, (channels, height, width), dtype=torch.uint8)
        pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())
        return tensor, pil_img

    def compareTensorToPIL(self, tensor, pil_image, msg=None):
        pil_tensor = torch.as_tensor(np.array(pil_image).transpose((2, 0, 1)))
        self.assertTrue(tensor.equal(pil_tensor), msg)

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

            max_diff = (colorsys_img - ft_hsv_img).abs().max()
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


if __name__ == '__main__':
    unittest.main()
