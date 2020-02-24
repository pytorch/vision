import os
import unittest
import sys

import torch
import torchvision
if sys.platform.startswith('linux'):
    from torchvision.io.image import read_jpeg, decode_jpeg
    from turbojpeg import TurboJPEG, TJPF_RGB
import numpy as np


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    for root, _, files in os.walk(directory):
        for fl in files:
            _, ext = os.path.splitext(fl)
            if ext == img_ext:
                yield os.path.join(root, fl)


def read_with_pyturbojpeg(img_path):
    jpeg = TurboJPEG()
    for img_path in get_images(IMAGE_DIR, ".jpg"):
        with open(img_path, 'rb') as f:
            return torch.from_numpy(jpeg.decode(f.read(), pixel_format=TJPF_RGB))


class ImageTester(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_read_jpeg(self):
        for img_path in get_images(IMAGE_DIR, ".jpg"):
            img_tjpeg = read_with_pyturbojpeg(img_path)
            img_ljpeg = read_jpeg(img_path)

            err = torch.abs(img_ljpeg.flatten().float() - img_tjpeg.flatten().float()).sum().float() / (img_ljpeg.shape[0] * img_ljpeg.shape[1] * img_ljpeg.shape[2] * 255)
            self.assertLessEqual(err, 1e-2)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Support only available on linux for now.")
    def test_decode_jpeg(self):
        for img_path in get_images(IMAGE_DIR, ".jpg"):
            img_tjpeg = read_with_pyturbojpeg(img_path)
            size = os.path.getsize(img_path)
            img_ljpeg = decode_jpeg(torch.from_file(img_path, dtype=torch.uint8, size=size))

            err = torch.abs(img_ljpeg.flatten().float() - img_tjpeg.flatten().float()).sum().float() / (img_ljpeg.shape[0] * img_ljpeg.shape[1] * img_ljpeg.shape[2] * 255)

            self.assertLessEqual(err, 1e-2)

        with self.assertRaisesRegex(ValueError, "Expected a non empty 1-dimensional tensor."):
            decode_jpeg(torch.empty((100, 1), dtype=torch.uint8))

        with self.assertRaisesRegex(ValueError, "Expected a torch.uint8 tensor."):
            decode_jpeg(torch.empty((100, ), dtype=torch.float16))

        with self.assertRaisesRegex(ValueError, "Invalid jpeg input."):
            decode_jpeg(torch.empty((100), dtype=torch.uint8))


if __name__ == '__main__':
    unittest.main()
