import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as F
from common_utils import assert_equal, needs_cuda
from PIL import __version__ as PILLOW_VERSION, Image
from torchvision.io.image import (
    _read_png_16,
    decode_image,
    decode_jpeg,
    decode_png,
    encode_jpeg,
    encode_png,
    ImageReadMode,
    read_file,
    read_image,
    write_file,
    write_jpeg,
    write_png,
)

IMAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FAKEDATA_DIR = os.path.join(IMAGE_ROOT, "fakedata")
IMAGE_DIR = os.path.join(FAKEDATA_DIR, "imagefolder")
DAMAGED_JPEG = os.path.join(IMAGE_ROOT, "damaged_jpeg")
DAMAGED_PNG = os.path.join(IMAGE_ROOT, "damaged_png")
ENCODE_JPEG = os.path.join(IMAGE_ROOT, "encode_jpeg")
INTERLACED_PNG = os.path.join(IMAGE_ROOT, "interlaced_png")
TOOSMALL_PNG = os.path.join(IMAGE_ROOT, "toosmall_png")
IS_WINDOWS = sys.platform in ("win32", "cygwin")
IS_MACOS = sys.platform == "darwin"
PILLOW_VERSION = tuple(int(x) for x in PILLOW_VERSION.split("."))


def _get_safe_image_name(name):
    # Used when we need to change the pytest "id" for an "image path" parameter.
    # If we don't, the test id (i.e. its name) will contain the whole path to the image, which is machine-specific,
    # and this creates issues when the test is running in a different machine than where it was collected
    # (typically, in fb internal infra)
    return name.split(os.path.sep)[-1]


def get_images(directory, img_ext):
    assert os.path.isdir(directory)
    image_paths = glob.glob(directory + f"/**/*{img_ext}", recursive=True)
    for path in image_paths:
        if path.split(os.sep)[-2] not in ["damaged_jpeg", "jpeg_write"]:
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


@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(IMAGE_ROOT, ".jpg")],
)
@pytest.mark.parametrize(
    "pil_mode, mode",
    [
        (None, ImageReadMode.UNCHANGED),
        ("L", ImageReadMode.GRAY),
        ("RGB", ImageReadMode.RGB),
    ],
)
def test_decode_jpeg(img_path, pil_mode, mode):

    with Image.open(img_path) as img:
        is_cmyk = img.mode == "CMYK"
        if pil_mode is not None:
            img = img.convert(pil_mode)
        img_pil = torch.from_numpy(np.array(img))
        if is_cmyk and mode == ImageReadMode.UNCHANGED:
            # flip the colors to match libjpeg
            img_pil = 255 - img_pil

    img_pil = normalize_dimensions(img_pil)
    data = read_file(img_path)
    img_ljpeg = decode_image(data, mode=mode)

    # Permit a small variation on pixel values to account for implementation
    # differences between Pillow and LibJPEG.
    abs_mean_diff = (img_ljpeg.type(torch.float32) - img_pil).abs().mean().item()
    assert abs_mean_diff < 2


def test_decode_jpeg_errors():
    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_jpeg(torch.empty((100, 1), dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="Expected a torch.uint8 tensor"):
        decode_jpeg(torch.empty((100,), dtype=torch.float16))

    with pytest.raises(RuntimeError, match="Not a JPEG file"):
        decode_jpeg(torch.empty((100), dtype=torch.uint8))


def test_decode_bad_huffman_images():
    # sanity check: make sure we can decode the bad Huffman encoding
    bad_huff = read_file(os.path.join(DAMAGED_JPEG, "bad_huffman.jpg"))
    decode_jpeg(bad_huff)


@pytest.mark.parametrize(
    "img_path",
    [
        pytest.param(truncated_image, id=_get_safe_image_name(truncated_image))
        for truncated_image in glob.glob(os.path.join(DAMAGED_JPEG, "corrupt*.jpg"))
    ],
)
def test_damaged_corrupt_images(img_path):
    # Truncated images should raise an exception
    data = read_file(img_path)
    if "corrupt34" in img_path:
        match_message = "Image is incomplete or truncated"
    else:
        match_message = "Unsupported marker type"
    with pytest.raises(RuntimeError, match=match_message):
        decode_jpeg(data)


@pytest.mark.parametrize(
    "img_path",
    [pytest.param(png_path, id=_get_safe_image_name(png_path)) for png_path in get_images(FAKEDATA_DIR, ".png")],
)
@pytest.mark.parametrize(
    "pil_mode, mode",
    [
        (None, ImageReadMode.UNCHANGED),
        ("L", ImageReadMode.GRAY),
        ("LA", ImageReadMode.GRAY_ALPHA),
        ("RGB", ImageReadMode.RGB),
        ("RGBA", ImageReadMode.RGB_ALPHA),
    ],
)
def test_decode_png(img_path, pil_mode, mode):

    with Image.open(img_path) as img:
        if pil_mode is not None:
            img = img.convert(pil_mode)
        img_pil = torch.from_numpy(np.array(img))

    img_pil = normalize_dimensions(img_pil)

    if img_path.endswith("16.png"):
        # 16 bits image decoding is supported, but only as a private API
        # FIXME: see https://github.com/pytorch/vision/issues/4731 for potential solutions to making it public
        with pytest.raises(RuntimeError, match="At most 8-bit PNG images are supported"):
            data = read_file(img_path)
            img_lpng = decode_image(data, mode=mode)

        img_lpng = _read_png_16(img_path, mode=mode)
        assert img_lpng.dtype == torch.int32
        # PIL converts 16 bits pngs in uint8
        img_lpng = torch.round(img_lpng / (2**16 - 1) * 255).to(torch.uint8)
    else:
        data = read_file(img_path)
        img_lpng = decode_image(data, mode=mode)

    tol = 0 if pil_mode is None else 1

    if PILLOW_VERSION >= (8, 3) and pil_mode == "LA":
        # Avoid checking the transparency channel until
        # https://github.com/python-pillow/Pillow/issues/5593#issuecomment-878244910
        # is fixed.
        # TODO: remove once fix is released in PIL. Should be > 8.3.1.
        img_lpng, img_pil = img_lpng[0], img_pil[0]

    torch.testing.assert_close(img_lpng, img_pil, atol=tol, rtol=0)


def test_decode_png_errors():
    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_png(torch.empty((), dtype=torch.uint8))
    with pytest.raises(RuntimeError, match="Content is not png"):
        decode_png(torch.randint(3, 5, (300,), dtype=torch.uint8))
    with pytest.raises(RuntimeError, match="Out of bound read in decode_png"):
        decode_png(read_file(os.path.join(DAMAGED_PNG, "sigsegv.png")))
    with pytest.raises(RuntimeError, match="Content is too small for png"):
        decode_png(read_file(os.path.join(TOOSMALL_PNG, "heapbof.png")))


@pytest.mark.parametrize(
    "img_path",
    [pytest.param(png_path, id=_get_safe_image_name(png_path)) for png_path in get_images(IMAGE_DIR, ".png")],
)
def test_encode_png(img_path):
    pil_image = Image.open(img_path)
    img_pil = torch.from_numpy(np.array(pil_image))
    img_pil = img_pil.permute(2, 0, 1)
    png_buf = encode_png(img_pil, compression_level=6)

    rec_img = Image.open(io.BytesIO(bytes(png_buf.tolist())))
    rec_img = torch.from_numpy(np.array(rec_img))
    rec_img = rec_img.permute(2, 0, 1)

    assert_equal(img_pil, rec_img)


def test_encode_png_errors():
    with pytest.raises(RuntimeError, match="Input tensor dtype should be uint8"):
        encode_png(torch.empty((3, 100, 100), dtype=torch.float32))

    with pytest.raises(RuntimeError, match="Compression level should be between 0 and 9"):
        encode_png(torch.empty((3, 100, 100), dtype=torch.uint8), compression_level=-1)

    with pytest.raises(RuntimeError, match="Compression level should be between 0 and 9"):
        encode_png(torch.empty((3, 100, 100), dtype=torch.uint8), compression_level=10)

    with pytest.raises(RuntimeError, match="The number of channels should be 1 or 3, got: 5"):
        encode_png(torch.empty((5, 100, 100), dtype=torch.uint8))


@pytest.mark.parametrize(
    "img_path",
    [pytest.param(png_path, id=_get_safe_image_name(png_path)) for png_path in get_images(IMAGE_DIR, ".png")],
)
def test_write_png(img_path, tmpdir):
    pil_image = Image.open(img_path)
    img_pil = torch.from_numpy(np.array(pil_image))
    img_pil = img_pil.permute(2, 0, 1)

    filename, _ = os.path.splitext(os.path.basename(img_path))
    torch_png = os.path.join(tmpdir, f"{filename}_torch.png")
    write_png(img_pil, torch_png, compression_level=6)
    saved_image = torch.from_numpy(np.array(Image.open(torch_png)))
    saved_image = saved_image.permute(2, 0, 1)

    assert_equal(img_pil, saved_image)


def test_read_file(tmpdir):
    fname, content = "test1.bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    with open(fpath, "wb") as f:
        f.write(content)

    data = read_file(fpath)
    expected = torch.tensor(list(content), dtype=torch.uint8)
    os.unlink(fpath)
    assert_equal(data, expected)

    with pytest.raises(RuntimeError, match="No such file or directory: 'tst'"):
        read_file("tst")


def test_read_file_non_ascii(tmpdir):
    fname, content = "日本語(Japanese).bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    with open(fpath, "wb") as f:
        f.write(content)

    data = read_file(fpath)
    expected = torch.tensor(list(content), dtype=torch.uint8)
    os.unlink(fpath)
    assert_equal(data, expected)


def test_write_file(tmpdir):
    fname, content = "test1.bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    content_tensor = torch.tensor(list(content), dtype=torch.uint8)
    write_file(fpath, content_tensor)

    with open(fpath, "rb") as f:
        saved_content = f.read()
    os.unlink(fpath)
    assert content == saved_content


def test_write_file_non_ascii(tmpdir):
    fname, content = "日本語(Japanese).bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    content_tensor = torch.tensor(list(content), dtype=torch.uint8)
    write_file(fpath, content_tensor)

    with open(fpath, "rb") as f:
        saved_content = f.read()
    os.unlink(fpath)
    assert content == saved_content


@pytest.mark.parametrize(
    "shape",
    [
        (27, 27),
        (60, 60),
        (105, 105),
    ],
)
def test_read_1_bit_png(shape, tmpdir):
    np_rng = np.random.RandomState(0)
    image_path = os.path.join(tmpdir, f"test_{shape}.png")
    pixels = np_rng.rand(*shape) > 0.5
    img = Image.fromarray(pixels)
    img.save(image_path)
    img1 = read_image(image_path)
    img2 = normalize_dimensions(torch.as_tensor(pixels * 255, dtype=torch.uint8))
    assert_equal(img1, img2)


@pytest.mark.parametrize(
    "shape",
    [
        (27, 27),
        (60, 60),
        (105, 105),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        ImageReadMode.UNCHANGED,
        ImageReadMode.GRAY,
    ],
)
def test_read_1_bit_png_consistency(shape, mode, tmpdir):
    np_rng = np.random.RandomState(0)
    image_path = os.path.join(tmpdir, f"test_{shape}.png")
    pixels = np_rng.rand(*shape) > 0.5
    img = Image.fromarray(pixels)
    img.save(image_path)
    img1 = read_image(image_path, mode)
    img2 = read_image(image_path, mode)
    assert_equal(img1, img2)


def test_read_interlaced_png():
    imgs = list(get_images(INTERLACED_PNG, ".png"))
    with Image.open(imgs[0]) as im1, Image.open(imgs[1]) as im2:
        assert not (im1.info.get("interlace") is im2.info.get("interlace"))
    img1 = read_image(imgs[0])
    img2 = read_image(imgs[1])
    assert_equal(img1, img2)


@needs_cuda
@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(IMAGE_ROOT, ".jpg")],
)
@pytest.mark.parametrize("mode", [ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB])
@pytest.mark.parametrize("scripted", (False, True))
def test_decode_jpeg_cuda(mode, img_path, scripted):
    if "cmyk" in img_path:
        pytest.xfail("Decoding a CMYK jpeg isn't supported")

    data = read_file(img_path)
    img = decode_image(data, mode=mode)
    f = torch.jit.script(decode_jpeg) if scripted else decode_jpeg
    img_nvjpeg = f(data, mode=mode, device="cuda")

    # Some difference expected between jpeg implementations
    assert (img.float() - img_nvjpeg.cpu().float()).abs().mean() < 2


@needs_cuda
def test_decode_image_cuda_raises():
    data = torch.randint(0, 127, size=(255,), device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        decode_image(data)


@needs_cuda
@pytest.mark.parametrize("cuda_device", ("cuda", "cuda:0", torch.device("cuda")))
def test_decode_jpeg_cuda_device_param(cuda_device):
    """Make sure we can pass a string or a torch.device as device param"""
    path = next(path for path in get_images(IMAGE_ROOT, ".jpg") if "cmyk" not in path)
    data = read_file(path)
    decode_jpeg(data, device=cuda_device)


@needs_cuda
def test_decode_jpeg_cuda_errors():
    data = read_file(next(get_images(IMAGE_ROOT, ".jpg")))
    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_jpeg(data.reshape(-1, 1), device="cuda")
    with pytest.raises(RuntimeError, match="input tensor must be on CPU"):
        decode_jpeg(data.to("cuda"), device="cuda")
    with pytest.raises(RuntimeError, match="Expected a torch.uint8 tensor"):
        decode_jpeg(data.to(torch.float), device="cuda")
    with pytest.raises(RuntimeError, match="Expected a cuda device"):
        torch.ops.image.decode_jpeg_cuda(data, ImageReadMode.UNCHANGED.value, "cpu")


def test_encode_jpeg_errors():

    with pytest.raises(RuntimeError, match="Input tensor dtype should be uint8"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.float32))

    with pytest.raises(ValueError, match="Image quality should be a positive number between 1 and 100"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=-1)

    with pytest.raises(ValueError, match="Image quality should be a positive number between 1 and 100"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.uint8), quality=101)

    with pytest.raises(RuntimeError, match="The number of channels should be 1 or 3, got: 5"):
        encode_jpeg(torch.empty((5, 100, 100), dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((1, 3, 100, 100), dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((100, 100), dtype=torch.uint8))


@pytest.mark.skipif(IS_MACOS, reason="https://github.com/pytorch/vision/issues/8031")
@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(ENCODE_JPEG, ".jpg")],
)
def test_encode_jpeg(img_path):
    img = read_image(img_path)

    pil_img = F.to_pil_image(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)

    encoded_jpeg_pil = torch.frombuffer(buf.getvalue(), dtype=torch.uint8)

    for src_img in [img, img.contiguous()]:
        encoded_jpeg_torch = encode_jpeg(src_img, quality=75)
        assert_equal(encoded_jpeg_torch, encoded_jpeg_pil)


@pytest.mark.skipif(IS_MACOS, reason="https://github.com/pytorch/vision/issues/8031")
@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(ENCODE_JPEG, ".jpg")],
)
def test_write_jpeg(img_path, tmpdir):
    tmpdir = Path(tmpdir)
    img = read_image(img_path)
    pil_img = F.to_pil_image(img)

    torch_jpeg = str(tmpdir / "torch.jpg")
    pil_jpeg = str(tmpdir / "pil.jpg")

    write_jpeg(img, torch_jpeg, quality=75)
    pil_img.save(pil_jpeg, quality=75)

    with open(torch_jpeg, "rb") as f:
        torch_bytes = f.read()

    with open(pil_jpeg, "rb") as f:
        pil_bytes = f.read()

    assert_equal(torch_bytes, pil_bytes)


if __name__ == "__main__":
    pytest.main([__file__])
