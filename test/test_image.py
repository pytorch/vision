import os
import io
import glob
import unittest
import sys

import torch
import torchvision
from PIL import Image
from torchvision.io.image import (
    read_png, decode_png, read_jpeg, decode_jpeg, encode_jpeg, write_jpeg)
import numpy as np

IMAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
IMAGE_DIR = os.path.join(IMAGE_ROOT, "fakedata", "imagefolder")
DAMAGED_JPEG = os.path.join(IMAGE_ROOT, 'damaged_jpeg')


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    for root, _, files in os.walk(directory):
        if os.path.basename(root) in {'damaged_jpeg', 'jpeg_write'}:
            continue

        for fl in files:
            _, ext = os.path.splitext(fl)
            if ext == img_ext:
                yield os.path.join(root, fl)


class ImageTester(unittest.TestCase):
    def test_read_jpeg(self):
        for img_path in get_images(IMAGE_ROOT, ".jpg"):
            img_pil = torch.load(img_path.replace('jpg', 'pth'))
            img_pil = img_pil.permute(2, 0, 1)
            img_ljpeg = read_jpeg(img_path)
            self.assertTrue(img_ljpeg.equal(img_pil))

    def test_decode_jpeg(self):
        for img_path in get_images(IMAGE_ROOT, ".jpg"):
            img_pil = torch.load(img_path.replace('jpg', 'pth'))
            img_pil = img_pil.permute(2, 0, 1)
            size = os.path.getsize(img_path)
            img_ljpeg = decode_jpeg(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(img_ljpeg.equal(img_pil))

        with self.assertRaisesRegex(ValueError, "Expected a non empty 1-dimensional tensor."):
            decode_jpeg(torch.empty((100, 1), dtype=torch.uint8))

        with self.assertRaisesRegex(ValueError, "Expected a torch.uint8 tensor."):
            decode_jpeg(torch.empty((100, ), dtype=torch.float16))

        with self.assertRaises(RuntimeError):
            decode_jpeg(torch.empty((100), dtype=torch.uint8))

    def test_damaged_images(self):
        # Test image with bad Huffman encoding (should not raise)
        bad_huff = os.path.join(DAMAGED_JPEG, 'bad_huffman.jpg')
        try:
            _ = read_jpeg(bad_huff)
        except RuntimeError:
            self.assertTrue(False)

        # Truncated images should raise an exception
        truncated_images = glob.glob(
            os.path.join(DAMAGED_JPEG, 'corrupt*.jpg'))
        for image_path in truncated_images:
            with self.assertRaises(RuntimeError):
                read_jpeg(image_path)

    def test_encode_jpeg(self):
        for img_path in get_images(IMAGE_ROOT, ".jpg"):
            dirname = os.path.dirname(img_path)
            filename, _ = os.path.splitext(os.path.basename(img_path))
            write_folder = os.path.join(dirname, 'jpeg_write')
            expected_file = os.path.join(
                write_folder, '{0}_pil.jpg'.format(filename))
            img = read_jpeg(img_path)

            with open(expected_file, 'rb') as f:
                pil_bytes = f.read()
                pil_bytes = torch.as_tensor(list(pil_bytes), dtype=torch.uint8)
            for src_img in [img, img.contiguous()]:
                # PIL sets jpeg quality to 75 by default
                jpeg_bytes = encode_jpeg(src_img, quality=75)
                self.assertTrue(jpeg_bytes.equal(pil_bytes))

        with self.assertRaisesRegex(
                RuntimeError, "Input tensor dtype should be uint8"):
            encode_jpeg(torch.empty((3, 100, 100), dtype=torch.float32))

        with self.assertRaisesRegex(
                ValueError, "Image quality should be a positive number "
                "between 1 and 100"):
            encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=-1)

        with self.assertRaisesRegex(
                ValueError, "Image quality should be a positive number "
                "between 1 and 100"):
            encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=101)

        with self.assertRaisesRegex(
                RuntimeError, "The number of channels should be 1 or 3, got: 5"):
            encode_jpeg(torch.empty((5, 100, 100), dtype=torch.uint8))

        with self.assertRaisesRegex(
                RuntimeError, "Input data should be a 3-dimensional tensor"):
            encode_jpeg(torch.empty((1, 3, 100, 100), dtype=torch.uint8))

        with self.assertRaisesRegex(
                RuntimeError, "Input data should be a 3-dimensional tensor"):
            encode_jpeg(torch.empty((100, 100), dtype=torch.uint8))

    def test_write_jpeg(self):
        for img_path in get_images(IMAGE_ROOT, ".jpg"):
            img = read_jpeg(img_path)

            basedir = os.path.dirname(img_path)
            filename, _ = os.path.splitext(os.path.basename(img_path))
            torch_jpeg = os.path.join(
                basedir, '{0}_torch.jpg'.format(filename))
            pil_jpeg = os.path.join(
                basedir, 'jpeg_write', '{0}_pil.jpg'.format(filename))

            write_jpeg(img, torch_jpeg, quality=75)

            with open(torch_jpeg, 'rb') as f:
                torch_bytes = f.read()

            with open(pil_jpeg, 'rb') as f:
                pil_bytes = f.read()

            os.remove(torch_jpeg)
            self.assertEqual(torch_bytes, pil_bytes)

    def test_read_png(self):
        # Check across .png
        for img_path in get_images(IMAGE_DIR, ".png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_pil = img_pil.permute(2, 0, 1)
            img_lpng = read_png(img_path)
            self.assertTrue(img_lpng.equal(img_pil))

    def test_decode_png(self):
        for img_path in get_images(IMAGE_DIR, ".png"):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_pil = img_pil.permute(2, 0, 1)
            size = os.path.getsize(img_path)
            img_lpng = decode_png(torch.from_file(img_path, dtype=torch.uint8, size=size))
            self.assertTrue(img_lpng.equal(img_pil))

            with self.assertRaises(ValueError):
                decode_png(torch.empty((), dtype=torch.uint8))
            with self.assertRaises(RuntimeError):
                decode_png(torch.randint(3, 5, (300,), dtype=torch.uint8))


if __name__ == '__main__':
    unittest.main()
