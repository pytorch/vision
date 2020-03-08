import os
import unittest
import sys

import torch
from PIL import Image
if sys.platform.startswith('linux'):
    from torchvision.io.image import read_png, decode_png
import numpy as np

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "imagefolder")


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    for root, dir, files in os.walk(directory):
        for fl in files:
            _, ext = os.path.splitext(fl)
            if ext == img_ext:
                yield os.path.join(root, fl)


class ImageTester(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_read_png(self):
        for img_path in get_images(IMAGE_DIR, ".png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_lpng = read_png(img_path)
            self.assertTrue(torch.equal(img_lpng, img_pil))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_decode_png(self):
        for img_path in get_images(IMAGE_DIR, ".png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            size = os.path.getsize(img_path)
            img_lpng = decode_png(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(torch.equal(img_lpng, img_pil))

        with self.assertRaisesRegex(ValueError, "Expected a non empty 1-dimensional tensor."):
            decode_png(torch.empty((100, 1), dtype=torch.uint8))

        with self.assertRaisesRegex(ValueError, "Expected a torch.uint8 tensor."):
            decode_png(torch.empty((100, ), dtype=torch.float16))

        with self.assertRaisesRegex(ValueError, "Invalid png input."):
            decode_png(torch.empty((100), dtype=torch.uint8))


if __name__ == '__main__':
    unittest.main()
