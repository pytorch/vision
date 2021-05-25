import glob
import io
import os
import sys
import unittest
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from common_utils import get_tmp_dir, needs_cuda, cpu_only
from _assert_utils import assert_equal

from torchvision.io.image import (
    decode_png, decode_jpeg, encode_jpeg, write_jpeg, decode_image, read_file,
    encode_png, write_png, write_file, ImageReadMode, read_image)

IMAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FAKEDATA_DIR = os.path.join(IMAGE_ROOT, "fakedata")
IMAGE_DIR = os.path.join(FAKEDATA_DIR, "imagefolder")
DAMAGED_JPEG = os.path.join(IMAGE_ROOT, 'damaged_jpeg')
ENCODE_JPEG = os.path.join(IMAGE_ROOT, "encode_jpeg")
IS_WINDOWS = sys.platform in ('win32', 'cygwin')


def _get_safe_image_name(name):
    # Used when we need to change the pytest "id" for an "image path" parameter.
    # If we don't, the test id (i.e. its name) will contain the whole path to the image, which is machine-specific,
    # and this creates issues when the test is running in a different machine than where it was collected
    # (typically, in fb internal infra)
    return name.split(os.path.sep)[-1]


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    image_paths = glob.glob(directory + f'/**/*{img_ext}', recursive=True)
    for path in image_paths:
        if path.split(os.sep)[-2] not in ['damaged_jpeg', 'jpeg_write']:
            yield path


def pil_read_image(img_path):
    with Image.open(img_path) as img:
        return torch.from_numpy(np.array(img))


def normalize_dimensions(img_pil):
    if len(img_pil.shape) == 3:
        img_pil = img_pil.permute(2, 0, 1)
    else:
        img_pil = img_pil.unsqueeze(0)
    return img_pil


class ImageTester(unittest.TestCase):
    def test_decode_jpeg(self):
        conversion = [(None, ImageReadMode.UNCHANGED), ("L", ImageReadMode.GRAY), ("RGB", ImageReadMode.RGB)]
        for img_path in get_images(IMAGE_ROOT, ".jpg"):
            for pil_mode, mode in conversion:
                with Image.open(img_path) as img:
                    is_cmyk = img.mode == "CMYK"
                    if pil_mode is not None:
                        if is_cmyk:
                            # libjpeg does not support the conversion
                            continue
                        img = img.convert(pil_mode)
                    img_pil = torch.from_numpy(np.array(img))
                    if is_cmyk:
                        # flip the colors to match libjpeg
                        img_pil = 255 - img_pil

                img_pil = normalize_dimensions(img_pil)
                data = read_file(img_path)
                img_ljpeg = decode_image(data, mode=mode)

                # Permit a small variation on pixel values to account for implementation
                # differences between Pillow and LibJPEG.
                abs_mean_diff = (img_ljpeg.type(torch.float32) - img_pil).abs().mean().item()
                self.assertTrue(abs_mean_diff < 2)

        with self.assertRaisesRegex(RuntimeError, "Expected a non empty 1-dimensional tensor"):
            decode_jpeg(torch.empty((100, 1), dtype=torch.uint8))

        with self.assertRaisesRegex(RuntimeError, "Expected a torch.uint8 tensor"):
            decode_jpeg(torch.empty((100,), dtype=torch.float16))

        with self.assertRaises(RuntimeError):
            decode_jpeg(torch.empty((100), dtype=torch.uint8))

    def test_damaged_images(self):
        # Test image with bad Huffman encoding (should not raise)
        bad_huff = read_file(os.path.join(DAMAGED_JPEG, 'bad_huffman.jpg'))
        try:
            _ = decode_jpeg(bad_huff)
        except RuntimeError:
            self.assertTrue(False)

        # Truncated images should raise an exception
        truncated_images = glob.glob(
            os.path.join(DAMAGED_JPEG, 'corrupt*.jpg'))
        for image_path in truncated_images:
            data = read_file(image_path)
            with self.assertRaises(RuntimeError):
                decode_jpeg(data)

    def test_decode_png(self):
        conversion = [(None, ImageReadMode.UNCHANGED), ("L", ImageReadMode.GRAY), ("LA", ImageReadMode.GRAY_ALPHA),
                      ("RGB", ImageReadMode.RGB), ("RGBA", ImageReadMode.RGB_ALPHA)]
        for img_path in get_images(FAKEDATA_DIR, ".png"):
            for pil_mode, mode in conversion:
                with Image.open(img_path) as img:
                    if pil_mode is not None:
                        img = img.convert(pil_mode)
                    img_pil = torch.from_numpy(np.array(img))

                img_pil = normalize_dimensions(img_pil)
                data = read_file(img_path)
                img_lpng = decode_image(data, mode=mode)

                tol = 0 if conversion is None else 1
                self.assertTrue(img_lpng.allclose(img_pil, atol=tol))

        with self.assertRaises(RuntimeError):
            decode_png(torch.empty((), dtype=torch.uint8))
        with self.assertRaises(RuntimeError):
            decode_png(torch.randint(3, 5, (300,), dtype=torch.uint8))

    def test_encode_png(self):
        for img_path in get_images(IMAGE_DIR, '.png'):
            pil_image = Image.open(img_path)
            img_pil = torch.from_numpy(np.array(pil_image))
            img_pil = img_pil.permute(2, 0, 1)
            png_buf = encode_png(img_pil, compression_level=6)

            rec_img = Image.open(io.BytesIO(bytes(png_buf.tolist())))
            rec_img = torch.from_numpy(np.array(rec_img))
            rec_img = rec_img.permute(2, 0, 1)

            assert_equal(img_pil, rec_img)

        with self.assertRaisesRegex(
                RuntimeError, "Input tensor dtype should be uint8"):
            encode_png(torch.empty((3, 100, 100), dtype=torch.float32))

        with self.assertRaisesRegex(
                RuntimeError, "Compression level should be between 0 and 9"):
            encode_png(torch.empty((3, 100, 100), dtype=torch.uint8),
                       compression_level=-1)

        with self.assertRaisesRegex(
                RuntimeError, "Compression level should be between 0 and 9"):
            encode_png(torch.empty((3, 100, 100), dtype=torch.uint8),
                       compression_level=10)

        with self.assertRaisesRegex(
                RuntimeError, "The number of channels should be 1 or 3, got: 5"):
            encode_png(torch.empty((5, 100, 100), dtype=torch.uint8))

    def test_write_png(self):
        with get_tmp_dir() as d:
            for img_path in get_images(IMAGE_DIR, '.png'):
                pil_image = Image.open(img_path)
                img_pil = torch.from_numpy(np.array(pil_image))
                img_pil = img_pil.permute(2, 0, 1)

                filename, _ = os.path.splitext(os.path.basename(img_path))
                torch_png = os.path.join(d, '{0}_torch.png'.format(filename))
                write_png(img_pil, torch_png, compression_level=6)
                saved_image = torch.from_numpy(np.array(Image.open(torch_png)))
                saved_image = saved_image.permute(2, 0, 1)

                assert_equal(img_pil, saved_image)

    def test_read_file(self):
        with get_tmp_dir() as d:
            fname, content = 'test1.bin', b'TorchVision\211\n'
            fpath = os.path.join(d, fname)
            with open(fpath, 'wb') as f:
                f.write(content)

            data = read_file(fpath)
            expected = torch.tensor(list(content), dtype=torch.uint8)
            assert_equal(data, expected)
            os.unlink(fpath)

        with self.assertRaisesRegex(
                RuntimeError, "No such file or directory: 'tst'"):
            read_file('tst')

    def test_read_file_non_ascii(self):
        with get_tmp_dir() as d:
            fname, content = '日本語(Japanese).bin', b'TorchVision\211\n'
            fpath = os.path.join(d, fname)
            with open(fpath, 'wb') as f:
                f.write(content)

            data = read_file(fpath)
            expected = torch.tensor(list(content), dtype=torch.uint8)
            assert_equal(data, expected)
            os.unlink(fpath)

    def test_write_file(self):
        with get_tmp_dir() as d:
            fname, content = 'test1.bin', b'TorchVision\211\n'
            fpath = os.path.join(d, fname)
            content_tensor = torch.tensor(list(content), dtype=torch.uint8)
            write_file(fpath, content_tensor)

            with open(fpath, 'rb') as f:
                saved_content = f.read()
            self.assertEqual(content, saved_content)
            os.unlink(fpath)

    def test_write_file_non_ascii(self):
        with get_tmp_dir() as d:
            fname, content = '日本語(Japanese).bin', b'TorchVision\211\n'
            fpath = os.path.join(d, fname)
            content_tensor = torch.tensor(list(content), dtype=torch.uint8)
            write_file(fpath, content_tensor)

            with open(fpath, 'rb') as f:
                saved_content = f.read()
            self.assertEqual(content, saved_content)
            os.unlink(fpath)


@needs_cuda
@pytest.mark.parametrize('img_path', [
    pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path))
    for jpeg_path in get_images(IMAGE_ROOT, ".jpg")
])
@pytest.mark.parametrize('mode', [ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB])
@pytest.mark.parametrize('scripted', (False, True))
def test_decode_jpeg_cuda(mode, img_path, scripted):
    if 'cmyk' in img_path:
        pytest.xfail("Decoding a CMYK jpeg isn't supported")
    tester = ImageTester()
    data = read_file(img_path)
    img = decode_image(data, mode=mode)
    f = torch.jit.script(decode_jpeg) if scripted else decode_jpeg
    img_nvjpeg = f(data, mode=mode, device='cuda')

    # Some difference expected between jpeg implementations
    tester.assertTrue((img.float() - img_nvjpeg.cpu().float()).abs().mean() < 2)


@needs_cuda
@pytest.mark.parametrize('cuda_device', ('cuda', 'cuda:0', torch.device('cuda')))
def test_decode_jpeg_cuda_device_param(cuda_device):
    """Make sure we can pass a string or a torch.device as device param"""
    data = read_file(next(get_images(IMAGE_ROOT, ".jpg")))
    decode_jpeg(data, device=cuda_device)


@needs_cuda
def test_decode_jpeg_cuda_errors():
    data = read_file(next(get_images(IMAGE_ROOT, ".jpg")))
    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_jpeg(data.reshape(-1, 1), device='cuda')
    with pytest.raises(RuntimeError, match="input tensor must be on CPU"):
        decode_jpeg(data.to('cuda'), device='cuda')
    with pytest.raises(RuntimeError, match="Expected a torch.uint8 tensor"):
        decode_jpeg(data.to(torch.float), device='cuda')
    with pytest.raises(RuntimeError, match="Expected a cuda device"):
        torch.ops.image.decode_jpeg_cuda(data, ImageReadMode.UNCHANGED.value, 'cpu')


@cpu_only
def test_encode_jpeg_errors():

    with pytest.raises(RuntimeError, match="Input tensor dtype should be uint8"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.float32))

    with pytest.raises(ValueError, match="Image quality should be a positive number "
                                         "between 1 and 100"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=-1)

    with pytest.raises(ValueError, match="Image quality should be a positive number "
                                         "between 1 and 100"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=101)

    with pytest.raises(RuntimeError, match="The number of channels should be 1 or 3, got: 5"):
        encode_jpeg(torch.empty((5, 100, 100), dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((1, 3, 100, 100), dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((100, 100), dtype=torch.uint8))


def _collect_if(cond):
    # TODO: remove this once test_encode_jpeg_windows and test_write_jpeg_windows
    # are removed
    def _inner(test_func):
        if cond:
            return test_func
        else:
            return pytest.mark.dont_collect(test_func)
    return _inner


@cpu_only
@_collect_if(cond=IS_WINDOWS)
def test_encode_jpeg_windows():
    # This test is *wrong*.
    # It compares a torchvision-encoded jpeg with a PIL-encoded jpeg, but it
    # starts encoding the torchvision version from an image that comes from
    # decode_jpeg, which can yield different results from pil.decode (see
    # test_decode... which uses a high tolerance).
    # Instead, we should start encoding from the exact same decoded image, for a
    # valid comparison. This is done in test_encode_jpeg, but unfortunately
    # these more correct tests fail on windows (probably because of a difference
    # in libjpeg) between torchvision and PIL.
    # FIXME: make the correct tests pass on windows and remove this.
    for img_path in get_images(ENCODE_JPEG, ".jpg"):
        dirname = os.path.dirname(img_path)
        filename, _ = os.path.splitext(os.path.basename(img_path))
        write_folder = os.path.join(dirname, 'jpeg_write')
        expected_file = os.path.join(
            write_folder, '{0}_pil.jpg'.format(filename))
        img = decode_jpeg(read_file(img_path))

        with open(expected_file, 'rb') as f:
            pil_bytes = f.read()
            pil_bytes = torch.as_tensor(list(pil_bytes), dtype=torch.uint8)
        for src_img in [img, img.contiguous()]:
            # PIL sets jpeg quality to 75 by default
            jpeg_bytes = encode_jpeg(src_img, quality=75)
            assert_equal(jpeg_bytes, pil_bytes)


@cpu_only
@_collect_if(cond=IS_WINDOWS)
def test_write_jpeg_windows():
    # FIXME: Remove this eventually, see test_encode_jpeg_windows
    with get_tmp_dir() as d:
        for img_path in get_images(ENCODE_JPEG, ".jpg"):
            data = read_file(img_path)
            img = decode_jpeg(data)

            basedir = os.path.dirname(img_path)
            filename, _ = os.path.splitext(os.path.basename(img_path))
            torch_jpeg = os.path.join(
                d, '{0}_torch.jpg'.format(filename))
            pil_jpeg = os.path.join(
                basedir, 'jpeg_write', '{0}_pil.jpg'.format(filename))

            write_jpeg(img, torch_jpeg, quality=75)

            with open(torch_jpeg, 'rb') as f:
                torch_bytes = f.read()

            with open(pil_jpeg, 'rb') as f:
                pil_bytes = f.read()

            assert_equal(torch_bytes, pil_bytes)


@cpu_only
@_collect_if(cond=not IS_WINDOWS)
@pytest.mark.parametrize('img_path', [
    pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path))
    for jpeg_path in get_images(ENCODE_JPEG, ".jpg")
])
def test_encode_jpeg(img_path):
    img = read_image(img_path)

    pil_img = F.to_pil_image(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=75)

    # pytorch can't read from raw bytes so we go through numpy
    pil_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    encoded_jpeg_pil = torch.as_tensor(pil_bytes)

    for src_img in [img, img.contiguous()]:
        encoded_jpeg_torch = encode_jpeg(src_img, quality=75)
        assert_equal(encoded_jpeg_torch, encoded_jpeg_pil)


@cpu_only
@_collect_if(cond=not IS_WINDOWS)
@pytest.mark.parametrize('img_path', [
    pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path))
    for jpeg_path in get_images(ENCODE_JPEG, ".jpg")
])
def test_write_jpeg(img_path):
    with get_tmp_dir() as d:
        d = Path(d)
        img = read_image(img_path)
        pil_img = F.to_pil_image(img)

        torch_jpeg = str(d / 'torch.jpg')
        pil_jpeg = str(d / 'pil.jpg')

        write_jpeg(img, torch_jpeg, quality=75)
        pil_img.save(pil_jpeg, quality=75)

        with open(torch_jpeg, 'rb') as f:
            torch_bytes = f.read()

        with open(pil_jpeg, 'rb') as f:
            pil_bytes = f.read()

        assert_equal(torch_bytes, pil_bytes)


if __name__ == '__main__':
    unittest.main()
