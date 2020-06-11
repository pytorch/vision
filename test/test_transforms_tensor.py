import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

import numpy as np

import unittest


class Tester(unittest.TestCase):
    def _create_data(self, height=3, width=3, channels=3):
        tensor = torch.randint(0, 255, (channels, height, width), dtype=torch.uint8)
        pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())
        return tensor, pil_img

    def compareTensorToPIL(self, tensor, pil_image):
        pil_tensor = torch.as_tensor(np.array(pil_image).transpose((2, 0, 1)))
        self.assertTrue(tensor.equal(pil_tensor))

    def _test_flip(self, func, method):
        tensor, pil_img = self._create_data()
        flip_tensor = getattr(F, func)(tensor)
        flip_pil_img = getattr(F, func)(pil_img)
        self.compareTensorToPIL(flip_tensor, flip_pil_img)

        scripted_fn = torch.jit.script(getattr(F, func))
        flip_tensor_script = scripted_fn(tensor)
        self.assertTrue(flip_tensor.equal(flip_tensor_script))

        # test for class interface
        f = getattr(T, method)()
        scripted_fn = torch.jit.script(f)
        scripted_fn(tensor)

    def test_random_horizontal_flip(self):
        self._test_flip('hflip', 'RandomHorizontalFlip')

    def test_random_vertical_flip(self):
        self._test_flip('vflip', 'RandomVerticalFlip')

    def test_adjustments(self):
        fns = ['adjust_brightness', 'adjust_contrast', 'adjust_saturation']
        for _ in range(20):
            factor = 3 * torch.rand(1).item()
            tensor, _ = self._create_data()
            pil_img = T.ToPILImage()(tensor)

            for func in fns:
                adjusted_tensor = getattr(F, func)(tensor, factor)
                adjusted_pil_img = getattr(F, func)(pil_img, factor)

                adjusted_pil_tensor = T.ToTensor()(adjusted_pil_img)
                scripted_fn = torch.jit.script(getattr(F, func))
                adjusted_tensor_script = scripted_fn(tensor, factor)

                if not tensor.dtype.is_floating_point:
                    adjusted_tensor = adjusted_tensor.to(torch.float) / 255
                    adjusted_tensor_script = adjusted_tensor_script.to(torch.float) / 255

                # F uses uint8 and F_t uses float, so there is a small
                # difference in values caused by (at most 5) truncations.
                max_diff = (adjusted_tensor - adjusted_pil_tensor).abs().max()
                max_diff_scripted = (adjusted_tensor - adjusted_tensor_script).abs().max()
                self.assertLess(max_diff, 5 / 255 + 1e-5)
                self.assertLess(max_diff_scripted, 5 / 255 + 1e-5)


if __name__ == '__main__':
    unittest.main()
