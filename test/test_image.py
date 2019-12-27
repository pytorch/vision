import os
import unittest

import torch
from PIL import Image
from torchvision.io.image import read_png, decode_png, decode_jpeg, read_jpeg
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
    def test_read_png(self):
        for img_path in get_images(IMAGE_DIR, "png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_lpng = read_png(img_path)
            self.assertTrue(torch.all(img_lpng == img_pil))

    def test_decode_png(self):
        for img_path in get_images(IMAGE_DIR, "png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            size = os.path.getsize(img_path)
            img_lpng = decode_png(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(torch.all(img_lpng == img_pil))

    def test_read_jpeg(self):
        for img_path in get_images(IMAGE_DIR, "jpg"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_ljpeg = read_jpeg(img_path)
            self.assertTrue(torch.all(img_ljpeg == img_pil))

    def test_decode_jpeg(self):
        for img_path in get_images(IMAGE_DIR, "jpg"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            size = os.path.getsize(img_path)
            img_ljpeg = decode_png(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(torch.all(img_ljpeg == img_pil))


if __name__ == '__main__':
    unittest.main()
