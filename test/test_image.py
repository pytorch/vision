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
    @unittest.skipUnless(sys.platform.startswith("linux") or sys.platform.startswith("darwin"), "Support only available on linux for now.")
    def test_read_png(self):
        for img_path in get_images(IMAGE_DIR, "png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_lpng = read_png(img_path)
            self.assertEqual(img_lpng, img_pil)

    @unittest.skipUnless(sys.platform.startswith("linux") or sys.platform.startswith("darwin"), "Support only available on linux for now.")
    def test_decode_png(self):
        for img_path in get_images(IMAGE_DIR, "png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            size = os.path.getsize(img_path)
            img_lpng = decode_png(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertEqual(img_lpng, img_pil)

            self.assertEqual(decode_png(torch.empty()), torch.empty())
            self.assertEqual(decode_png(torch.randint(3, 5, (300,))), torch.empty())


if __name__ == '__main__':
    unittest.main()
