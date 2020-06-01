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

    def test_random_horizontal_flip(self):
        tensor, pil_img = self._create_data()
        flip_tensor = F.hflip(tensor)
        flip_pil_img = F.hflip(pil_img)
        self.compareTensorToPIL(flip_tensor, flip_pil_img)

        scripted_fn = torch.jit.script(F.hflip)
        flip_tensor_script = scripted_fn(tensor)
        self.assertTrue(flip_tensor.equal(flip_tensor_script))

        # test for class interface
        f = T.RandomHorizontalFlip()
        scripted_fn = torch.jit.script(f)
        scripted_fn(tensor)


if __name__ == '__main__':
    unittest.main()
