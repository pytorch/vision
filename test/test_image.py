import os
import unittest
import sys

import torch
import torchvision
from PIL import Image
if sys.platform.startswith('linux'):
    from torchvision.io.image import read_jpeg, decode_jpeg
import numpy as np


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    for root, _, files in os.walk(directory):
        for fl in files:
            _, ext = os.path.splitext(fl)
            if ext == img_ext:
                yield os.path.join(root, fl)


class ImageTester(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_read_jpeg(self):
        for img_path in get_images(IMAGE_DIR, ".jpg"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_ljpeg = read_jpeg(img_path)
            self.assertEqual(img_ljpeg, img_pil)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_decode_jpeg(self):
        for img_path in get_images(IMAGE_DIR, ".jpg"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            size = os.path.getsize(img_path)
            img_ljpeg = decode_jpeg(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(img_ljpeg, img_pil)

        self.assertEqual(decode_jpeg(torch.empty((1,100), dtype=torch.uint8)), None)


if __name__ == '__main__':
    unittest.main()
