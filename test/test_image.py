import concurrent.futures
import glob
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
import torchvision.transforms.v2.functional as F
from common_utils import assert_equal, cpu_and_cuda, IN_OSS_CI, needs_cuda
from PIL import __version__ as PILLOW_VERSION, Image, ImageOps, ImageSequence
from torchvision.io.image import (
    decode_avif,
    decode_gif,
    decode_heic,
    decode_image,
    decode_jpeg,
    decode_png,
    decode_webp,
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
IS_LINUX = sys.platform == "linux"
PILLOW_VERSION = tuple(int(x) for x in PILLOW_VERSION.split("."))
WEBP_TEST_IMAGES_DIR = os.environ.get("WEBP_TEST_IMAGES_DIR", "")
# See https://github.com/pytorch/vision/pull/8724#issuecomment-2503964558
HEIC_AVIF_MESSAGE = "AVIF and HEIF only available on linux."


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
@pytest.mark.parametrize("scripted", (False, True))
@pytest.mark.parametrize("decode_fun", (decode_jpeg, decode_image))
def test_decode_jpeg(img_path, pil_mode, mode, scripted, decode_fun):

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
    if scripted:
        decode_fun = torch.jit.script(decode_fun)
    img_ljpeg = decode_fun(data, mode=mode)

    # Permit a small variation on pixel values to account for implementation
    # differences between Pillow and LibJPEG.
    abs_mean_diff = (img_ljpeg.type(torch.float32) - img_pil).abs().mean().item()
    assert abs_mean_diff < 2


@pytest.mark.parametrize("codec", ["png", "jpeg"])
@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8, 0])
def test_decode_with_exif_orientation(tmpdir, codec, orientation):
    fp = os.path.join(tmpdir, f"exif_oriented_{orientation}.{codec}")
    t = torch.randint(0, 256, size=(3, 256, 257), dtype=torch.uint8)
    im = F.to_pil_image(t)
    exif = im.getexif()
    exif[0x0112] = orientation  # set exif orientation
    im.save(fp, codec.upper(), exif=exif.tobytes())

    data = read_file(fp)
    output = decode_image(data, apply_exif_orientation=True)

    pimg = Image.open(fp)
    pimg = ImageOps.exif_transpose(pimg)

    expected = F.pil_to_tensor(pimg)
    torch.testing.assert_close(expected, output)


@pytest.mark.parametrize("size", [65533, 1, 7, 10, 23, 33])
def test_invalid_exif(tmpdir, size):
    # Inspired from a PIL test:
    # https://github.com/python-pillow/Pillow/blob/8f63748e50378424628155994efd7e0739a4d1d1/Tests/test_file_jpeg.py#L299
    fp = os.path.join(tmpdir, "invalid_exif.jpg")
    t = torch.randint(0, 256, size=(3, 256, 257), dtype=torch.uint8)
    im = F.to_pil_image(t)
    im.save(fp, "JPEG", exif=b"1" * size)

    data = read_file(fp)
    output = decode_image(data, apply_exif_orientation=True)

    pimg = Image.open(fp)
    pimg = ImageOps.exif_transpose(pimg)

    expected = F.pil_to_tensor(pimg)
    torch.testing.assert_close(expected, output)


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
@pytest.mark.parametrize("scripted", (False, True))
@pytest.mark.parametrize("decode_fun", (decode_png, decode_image))
def test_decode_png(img_path, pil_mode, mode, scripted, decode_fun):

    if scripted:
        decode_fun = torch.jit.script(decode_fun)

    with Image.open(img_path) as img:
        if pil_mode is not None:
            img = img.convert(pil_mode)
        img_pil = torch.from_numpy(np.array(img))

    img_pil = normalize_dimensions(img_pil)

    if img_path.endswith("16.png"):
        data = read_file(img_path)
        img_lpng = decode_fun(data, mode=mode)
        assert img_lpng.dtype == torch.uint16
        # PIL converts 16 bits pngs to uint8
        img_lpng = F.to_dtype(img_lpng, torch.uint8, scale=True)
    else:
        data = read_file(img_path)
        img_lpng = decode_fun(data, mode=mode)

    tol = 0 if pil_mode is None else 1

    if PILLOW_VERSION >= (8, 3) and pil_mode == "LA":
        # Avoid checking the transparency channel until
        # https://github.com/python-pillow/Pillow/issues/5593#issuecomment-878244910
        # is fixed.
        # TODO: remove once fix is released in PIL. Should be > 8.3.1.
        img_lpng, img_pil = img_lpng[0], img_pil[0]

    torch.testing.assert_close(img_lpng, img_pil, atol=tol, rtol=0)


def test_decode_png_errors():
    with pytest.raises(RuntimeError, match="Out of bound read in decode_png"):
        decode_png(read_file(os.path.join(DAMAGED_PNG, "sigsegv.png")))
    with pytest.raises(RuntimeError, match="Content is too small for png"):
        decode_png(read_file(os.path.join(TOOSMALL_PNG, "heapbof.png")))


@pytest.mark.parametrize(
    "img_path",
    [pytest.param(png_path, id=_get_safe_image_name(png_path)) for png_path in get_images(IMAGE_DIR, ".png")],
)
@pytest.mark.parametrize("scripted", (True, False))
def test_encode_png(img_path, scripted):
    pil_image = Image.open(img_path)
    img_pil = torch.from_numpy(np.array(pil_image))
    img_pil = img_pil.permute(2, 0, 1)
    encode = torch.jit.script(encode_png) if scripted else encode_png
    png_buf = encode(img_pil, compression_level=6)

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
@pytest.mark.parametrize("scripted", (True, False))
def test_write_png(img_path, tmpdir, scripted):
    pil_image = Image.open(img_path)
    img_pil = torch.from_numpy(np.array(pil_image))
    img_pil = img_pil.permute(2, 0, 1)

    filename, _ = os.path.splitext(os.path.basename(img_path))
    torch_png = os.path.join(tmpdir, f"{filename}_torch.png")
    write = torch.jit.script(write_png) if scripted else write_png
    write(img_pil, torch_png, compression_level=6)
    saved_image = torch.from_numpy(np.array(Image.open(torch_png)))
    saved_image = saved_image.permute(2, 0, 1)

    assert_equal(img_pil, saved_image)


def test_read_image():
    # Just testing torchcsript, the functionality is somewhat tested already in other tests.
    path = next(get_images(IMAGE_ROOT, ".jpg"))
    out = read_image(path)
    out_scripted = torch.jit.script(read_image)(path)
    torch.testing.assert_close(out, out_scripted, atol=0, rtol=0)


@pytest.mark.parametrize("scripted", (True, False))
def test_read_file(tmpdir, scripted):
    fname, content = "test1.bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    with open(fpath, "wb") as f:
        f.write(content)

    fun = torch.jit.script(read_file) if scripted else read_file
    data = fun(fpath)
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


@pytest.mark.parametrize("scripted", (True, False))
def test_write_file(tmpdir, scripted):
    fname, content = "test1.bin", b"TorchVision\211\n"
    fpath = os.path.join(tmpdir, fname)
    content_tensor = torch.tensor(list(content), dtype=torch.uint8)
    write = torch.jit.script(write_file) if scripted else write_file
    write(fpath, content_tensor)

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
@pytest.mark.parametrize("mode", [ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB])
@pytest.mark.parametrize("scripted", (False, True))
def test_decode_jpegs_cuda(mode, scripted):
    encoded_images = []
    for jpeg_path in get_images(IMAGE_ROOT, ".jpg"):
        if "cmyk" in jpeg_path:
            continue
        encoded_image = read_file(jpeg_path)
        encoded_images.append(encoded_image)
    decoded_images_cpu = decode_jpeg(encoded_images, mode=mode)
    decode_fn = torch.jit.script(decode_jpeg) if scripted else decode_jpeg

    # test multithreaded decoding
    # in the current version we prevent this by using a lock but we still want to test it
    num_workers = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(decode_fn, encoded_images, mode, "cuda") for _ in range(num_workers)]
    decoded_images_threaded = [future.result() for future in futures]
    assert len(decoded_images_threaded) == num_workers
    for decoded_images in decoded_images_threaded:
        assert len(decoded_images) == len(encoded_images)
        for decoded_image_cuda, decoded_image_cpu in zip(decoded_images, decoded_images_cpu):
            assert decoded_image_cuda.shape == decoded_image_cpu.shape
            assert decoded_image_cuda.dtype == decoded_image_cpu.dtype == torch.uint8
            assert (decoded_image_cuda.cpu().float() - decoded_image_cpu.cpu().float()).abs().mean() < 2


@needs_cuda
def test_decode_image_cuda_raises():
    data = torch.randint(0, 127, size=(255,), device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        decode_image(data)


@needs_cuda
def test_decode_jpeg_cuda_device_param():
    path = next(path for path in get_images(IMAGE_ROOT, ".jpg") if "cmyk" not in path)
    data = read_file(path)
    current_device = torch.cuda.current_device()
    current_stream = torch.cuda.current_stream()
    num_devices = torch.cuda.device_count()
    devices = ["cuda", torch.device("cuda")] + [torch.device(f"cuda:{i}") for i in range(num_devices)]
    results = []
    for device in devices:
        results.append(decode_jpeg(data, device=device))
    assert len(results) == len(devices)
    for result in results:
        assert torch.all(result.cpu() == results[0].cpu())
    assert current_device == torch.cuda.current_device()
    assert current_stream == torch.cuda.current_stream()


@needs_cuda
def test_decode_jpeg_cuda_errors():
    data = read_file(next(get_images(IMAGE_ROOT, ".jpg")))
    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_jpeg(data.reshape(-1, 1), device="cuda")
    with pytest.raises(ValueError, match="must be tensors"):
        decode_jpeg([1, 2, 3])
    with pytest.raises(ValueError, match="Input tensor must be a CPU tensor"):
        decode_jpeg(data.to("cuda"), device="cuda")
    with pytest.raises(RuntimeError, match="Expected a torch.uint8 tensor"):
        decode_jpeg(data.to(torch.float), device="cuda")
    with pytest.raises(RuntimeError, match="Expected the device parameter to be a cuda device"):
        torch.ops.image.decode_jpegs_cuda([data], ImageReadMode.UNCHANGED.value, "cpu")
    with pytest.raises(ValueError, match="Input tensor must be a CPU tensor"):
        decode_jpeg(
            torch.empty((100,), dtype=torch.uint8, device="cuda"),
        )
    with pytest.raises(ValueError, match="Input list must contain tensors on CPU"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8, device="cuda"),
                torch.empty((100,), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(ValueError, match="Input list must contain tensors on CPU"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8, device="cuda"),
                torch.empty((100,), dtype=torch.uint8, device="cuda"),
            ],
            device="cuda",
        )

    with pytest.raises(ValueError, match="Input list must contain tensors on CPU"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8, device="cpu"),
                torch.empty((100,), dtype=torch.uint8, device="cuda"),
            ],
            device="cuda",
        )

    with pytest.raises(RuntimeError, match="Expected a torch.uint8 tensor"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8),
                torch.empty((100,), dtype=torch.float32),
            ],
            device="cuda",
        )

    with pytest.raises(RuntimeError, match="Expected a non empty 1-dimensional tensor"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8),
                torch.empty((1, 100), dtype=torch.uint8),
            ],
            device="cuda",
        )

    with pytest.raises(RuntimeError, match="Error while decoding JPEG images"):
        decode_jpeg(
            [
                torch.empty((100,), dtype=torch.uint8),
                torch.empty((100,), dtype=torch.uint8),
            ],
            device="cuda",
        )

    with pytest.raises(ValueError, match="Input list must contain at least one element"):
        decode_jpeg([], device="cuda")


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
@pytest.mark.parametrize("scripted", (True, False))
def test_encode_jpeg(img_path, scripted):
    img = read_image(img_path)

    pil_img = F.to_pil_image(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)

    encoded_jpeg_pil = torch.frombuffer(buf.getvalue(), dtype=torch.uint8)

    encode = torch.jit.script(encode_jpeg) if scripted else encode_jpeg
    for src_img in [img, img.contiguous()]:
        encoded_jpeg_torch = encode(src_img, quality=75)
        assert_equal(encoded_jpeg_torch, encoded_jpeg_pil)


@needs_cuda
def test_encode_jpeg_cuda_device_param():
    path = next(path for path in get_images(IMAGE_ROOT, ".jpg") if "cmyk" not in path)

    data = read_image(path)

    current_device = torch.cuda.current_device()
    current_stream = torch.cuda.current_stream()
    num_devices = torch.cuda.device_count()
    devices = ["cuda", torch.device("cuda")] + [torch.device(f"cuda:{i}") for i in range(num_devices)]
    results = []
    for device in devices:
        results.append(encode_jpeg(data.to(device=device)))
    assert len(results) == len(devices)
    for result in results:
        assert torch.all(result.cpu() == results[0].cpu())
    assert current_device == torch.cuda.current_device()
    assert current_stream == torch.cuda.current_stream()


@needs_cuda
@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(IMAGE_ROOT, ".jpg")],
)
@pytest.mark.parametrize("scripted", (False, True))
@pytest.mark.parametrize("contiguous", (False, True))
def test_encode_jpeg_cuda(img_path, scripted, contiguous):
    decoded_image_tv = read_image(img_path)
    encode_fn = torch.jit.script(encode_jpeg) if scripted else encode_jpeg

    if "cmyk" in img_path:
        pytest.xfail("Encoding a CMYK jpeg isn't supported")
    if decoded_image_tv.shape[0] == 1:
        pytest.xfail("Decoding a grayscale jpeg isn't supported")
        # For more detail as to why check out: https://github.com/NVIDIA/cuda-samples/issues/23#issuecomment-559283013
    if contiguous:
        decoded_image_tv = decoded_image_tv[None].contiguous(memory_format=torch.contiguous_format)[0]
    else:
        decoded_image_tv = decoded_image_tv[None].contiguous(memory_format=torch.channels_last)[0]
    encoded_jpeg_cuda_tv = encode_fn(decoded_image_tv.cuda(), quality=75)
    decoded_jpeg_cuda_tv = decode_jpeg(encoded_jpeg_cuda_tv.cpu())

    # the actual encoded bytestreams from libnvjpeg and libjpeg-turbo differ for the same quality
    # instead, we re-decode the encoded image and compare to the original
    abs_mean_diff = (decoded_jpeg_cuda_tv.float() - decoded_image_tv.float()).abs().mean().item()
    assert abs_mean_diff < 3


@needs_cuda
def test_encode_jpeg_cuda_sync():
    """
    Non-regression test for https://github.com/pytorch/vision/issues/8587.
    Attempts to reproduce an intermittent CUDA stream synchronization bug
    by randomly creating images and round-tripping them via encode_jpeg
    and decode_jpeg on the GPU. Fails if the mean difference in uint8 range
    exceeds 5.
    """
    torch.manual_seed(42)

    # manual testing shows this bug appearing often in iterations between 50 and 100
    # as a synchronization bug, this can't be reliably reproduced
    max_iterations = 100
    threshold = 5.0  # in [0..255]

    device = torch.device("cuda")

    for iteration in range(max_iterations):
        height, width = torch.randint(4000, 5000, size=(2,))

        image = torch.linspace(0, 1, steps=height * width, device=device)
        image = image.view(1, height, width).expand(3, -1, -1)

        image = (image * 255).clamp(0, 255).to(torch.uint8)
        jpeg_bytes = encode_jpeg(image, quality=100)

        decoded_image = decode_jpeg(jpeg_bytes.cpu(), device=device)
        mean_difference = (image.float() - decoded_image.float()).abs().mean().item()

        assert mean_difference <= threshold, (
            f"Encode/decode mismatch at iteration={iteration}, "
            f"size={height}x{width}, mean diff={mean_difference:.2f}"
        )


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("scripted", (True, False))
@pytest.mark.parametrize("contiguous", (True, False))
def test_encode_jpegs_batch(scripted, contiguous, device):
    if device == "cpu" and IS_MACOS:
        pytest.skip("https://github.com/pytorch/vision/issues/8031")
    decoded_images_tv = []
    for jpeg_path in get_images(IMAGE_ROOT, ".jpg"):
        if "cmyk" in jpeg_path:
            continue
        decoded_image = read_image(jpeg_path)
        if decoded_image.shape[0] == 1:
            continue
        if contiguous:
            decoded_image = decoded_image[None].contiguous(memory_format=torch.contiguous_format)[0]
        else:
            decoded_image = decoded_image[None].contiguous(memory_format=torch.channels_last)[0]
        decoded_images_tv.append(decoded_image)

    encode_fn = torch.jit.script(encode_jpeg) if scripted else encode_jpeg

    decoded_images_tv_device = [img.to(device=device) for img in decoded_images_tv]
    encoded_jpegs_tv_device = encode_fn(decoded_images_tv_device, quality=75)
    encoded_jpegs_tv_device = [decode_jpeg(img.cpu()) for img in encoded_jpegs_tv_device]

    for original, encoded_decoded in zip(decoded_images_tv, encoded_jpegs_tv_device):
        c, h, w = original.shape
        abs_mean_diff = (original.float() - encoded_decoded.float()).abs().mean().item()
        assert abs_mean_diff < 3

    # test multithreaded decoding
    # in the current version we prevent this by using a lock but we still want to test it
    num_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(encode_fn, decoded_images_tv_device) for _ in range(num_workers)]
    encoded_images_threaded = [future.result() for future in futures]
    assert len(encoded_images_threaded) == num_workers
    for encoded_images in encoded_images_threaded:
        assert len(decoded_images_tv_device) == len(encoded_images)
        for i, (encoded_image_cuda, decoded_image_tv) in enumerate(zip(encoded_images, decoded_images_tv_device)):
            # make sure all the threads produce identical outputs
            assert torch.all(encoded_image_cuda == encoded_images_threaded[0][i])

            # make sure the outputs are identical or close enough to baseline
            decoded_cuda_encoded_image = decode_jpeg(encoded_image_cuda.cpu())
            assert decoded_cuda_encoded_image.shape == decoded_image_tv.shape
            assert decoded_cuda_encoded_image.dtype == decoded_image_tv.dtype
            assert (decoded_cuda_encoded_image.cpu().float() - decoded_image_tv.cpu().float()).abs().mean() < 3


@needs_cuda
def test_single_encode_jpeg_cuda_errors():
    with pytest.raises(RuntimeError, match="Input tensor dtype should be uint8"):
        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.float32, device="cuda"))

    with pytest.raises(RuntimeError, match="The number of channels should be 3, got: 5"):
        encode_jpeg(torch.empty((5, 100, 100), dtype=torch.uint8, device="cuda"))

    with pytest.raises(RuntimeError, match="The number of channels should be 3, got: 1"):
        encode_jpeg(torch.empty((1, 100, 100), dtype=torch.uint8, device="cuda"))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((1, 3, 100, 100), dtype=torch.uint8, device="cuda"))

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(torch.empty((100, 100), dtype=torch.uint8, device="cuda"))


@needs_cuda
def test_batch_encode_jpegs_cuda_errors():
    with pytest.raises(RuntimeError, match="Input tensor dtype should be uint8"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((3, 100, 100), dtype=torch.float32, device="cuda"),
            ]
        )

    with pytest.raises(RuntimeError, match="The number of channels should be 3, got: 5"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((5, 100, 100), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(RuntimeError, match="The number of channels should be 3, got: 1"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((1, 100, 100), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((1, 3, 100, 100), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(RuntimeError, match="Input data should be a 3-dimensional tensor"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((100, 100), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(RuntimeError, match="Input tensor should be on CPU"):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cpu"),
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
            ]
        )

    with pytest.raises(
        RuntimeError, match="All input tensors must be on the same CUDA device when encoding with nvjpeg"
    ):
        encode_jpeg(
            [
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
                torch.empty((3, 100, 100), dtype=torch.uint8, device="cpu"),
            ]
        )

    if torch.cuda.device_count() >= 2:
        with pytest.raises(
            RuntimeError, match="All input tensors must be on the same CUDA device when encoding with nvjpeg"
        ):
            encode_jpeg(
                [
                    torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda:0"),
                    torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda:1"),
                ]
            )

    with pytest.raises(ValueError, match="encode_jpeg requires at least one input tensor when a list is passed"):
        encode_jpeg([])


@pytest.mark.skipif(IS_MACOS, reason="https://github.com/pytorch/vision/issues/8031")
@pytest.mark.parametrize(
    "img_path",
    [pytest.param(jpeg_path, id=_get_safe_image_name(jpeg_path)) for jpeg_path in get_images(ENCODE_JPEG, ".jpg")],
)
@pytest.mark.parametrize("scripted", (True, False))
def test_write_jpeg(img_path, tmpdir, scripted):
    tmpdir = Path(tmpdir)
    img = read_image(img_path)
    pil_img = F.to_pil_image(img)

    torch_jpeg = str(tmpdir / "torch.jpg")
    pil_jpeg = str(tmpdir / "pil.jpg")

    write = torch.jit.script(write_jpeg) if scripted else write_jpeg
    write(img, torch_jpeg, quality=75)
    pil_img.save(pil_jpeg, quality=75)

    with open(torch_jpeg, "rb") as f:
        torch_bytes = f.read()

    with open(pil_jpeg, "rb") as f:
        pil_bytes = f.read()

    assert_equal(torch_bytes, pil_bytes)


def test_pathlib_support(tmpdir):
    # Just make sure pathlib.Path is supported where relevant

    jpeg_path = Path(next(get_images(ENCODE_JPEG, ".jpg")))

    read_file(jpeg_path)
    read_image(jpeg_path)

    write_path = Path(tmpdir) / "whatever"
    img = torch.randint(0, 10, size=(3, 4, 4), dtype=torch.uint8)

    write_file(write_path, data=img.flatten())
    write_jpeg(img, write_path)
    write_png(img, write_path)


@pytest.mark.parametrize(
    "name", ("gifgrid", "fire", "porsche", "treescap", "treescap-interlaced", "solid2", "x-trans", "earth")
)
@pytest.mark.parametrize("scripted", (True, False))
def test_decode_gif(tmpdir, name, scripted):
    # Using test images from GIFLIB
    # https://sourceforge.net/p/giflib/code/ci/master/tree/pic/, we assert PIL
    # and torchvision decoded outputs are equal.
    # We're not testing against "welcome2" because PIL and GIFLIB disagee on what
    # the background color should be (likely a difference in the way they handle
    # transparency?)
    # 'earth' image is from wikipedia, licensed under CC BY-SA 3.0
    # https://creativecommons.org/licenses/by-sa/3.0/
    # it allows to properly test for transparency, TOP-LEFT offsets, and
    # disposal modes.

    path = tmpdir / f"{name}.gif"
    if name == "earth":
        if IN_OSS_CI:
            # TODO: Fix this... one day.
            pytest.skip("Skipping 'earth' test as it's flaky on OSS CI")
        url = "https://upload.wikimedia.org/wikipedia/commons/2/2c/Rotating_earth_%28large%29.gif"
    else:
        url = f"https://sourceforge.net/p/giflib/code/ci/master/tree/pic/{name}.gif?format=raw"
    with open(path, "wb") as f:
        f.write(requests.get(url).content)

    encoded_bytes = read_file(path)
    f = torch.jit.script(decode_gif) if scripted else decode_gif
    tv_out = f(encoded_bytes)
    if tv_out.ndim == 3:
        tv_out = tv_out[None]

    assert tv_out.is_contiguous(memory_format=torch.channels_last)

    # For some reason, not using Image.open() as a CM causes "ResourceWarning: unclosed file"
    with Image.open(path) as pil_img:
        pil_seq = ImageSequence.Iterator(pil_img)

        for pil_frame, tv_frame in zip(pil_seq, tv_out):
            pil_frame = F.pil_to_tensor(pil_frame.convert("RGB"))
            torch.testing.assert_close(tv_frame, pil_frame, atol=0, rtol=0)


@pytest.mark.parametrize(
    "decode_fun, match",
    [
        (decode_png, "Content is not png"),
        (decode_jpeg, "Not a JPEG file"),
        (decode_gif, re.escape("DGifOpenFileName() failed - 103")),
        (decode_webp, "WebPGetFeatures failed."),
        pytest.param(
            decode_avif, "BMFF parsing failed", marks=pytest.mark.skipif(not IS_LINUX, reason=HEIC_AVIF_MESSAGE)
        ),
        pytest.param(
            decode_heic,
            "Invalid input: No 'ftyp' box",
            marks=pytest.mark.skipif(not IS_LINUX, reason=HEIC_AVIF_MESSAGE),
        ),
    ],
)
def test_decode_bad_encoded_data(decode_fun, match):
    encoded_data = torch.randint(0, 256, (100,), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="Input tensor must be 1-dimensional"):
        decode_fun(encoded_data[None])
    with pytest.raises(RuntimeError, match="Input tensor must have uint8 data type"):
        decode_fun(encoded_data.float())
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        decode_fun(encoded_data[::2])
    with pytest.raises(RuntimeError, match=match):
        decode_fun(encoded_data)


@pytest.mark.parametrize("decode_fun", (decode_webp, decode_image))
@pytest.mark.parametrize("scripted", (False, True))
def test_decode_webp(decode_fun, scripted):
    encoded_bytes = read_file(next(get_images(FAKEDATA_DIR, ".webp")))
    if scripted:
        decode_fun = torch.jit.script(decode_fun)
    img = decode_fun(encoded_bytes)
    assert img.shape == (3, 100, 100)
    assert img[None].is_contiguous(memory_format=torch.channels_last)
    img += 123  # make sure image buffer wasn't freed by underlying decoding lib


# This test is skipped by default because it requires webp images that we're not
# including within the repo. The test images were downloaded manually from the
# different pages of https://developers.google.com/speed/webp/gallery
@pytest.mark.skipif(not WEBP_TEST_IMAGES_DIR, reason="WEBP_TEST_IMAGES_DIR is not set")
@pytest.mark.parametrize("decode_fun", (decode_webp, decode_image))
@pytest.mark.parametrize("scripted", (False, True))
@pytest.mark.parametrize(
    "mode, pil_mode",
    (
        # Note that converting an RGBA image to RGB leads to bad results because the
        # transparent pixels aren't necessarily set to "black" or "white", they can be
        # random stuff. This is consistent with PIL results.
        (ImageReadMode.RGB, "RGB"),
        (ImageReadMode.RGB_ALPHA, "RGBA"),
        (ImageReadMode.UNCHANGED, None),
    ),
)
@pytest.mark.parametrize("filename", Path(WEBP_TEST_IMAGES_DIR).glob("*.webp"), ids=lambda p: p.name)
def test_decode_webp_against_pil(decode_fun, scripted, mode, pil_mode, filename):
    encoded_bytes = read_file(filename)
    if scripted:
        decode_fun = torch.jit.script(decode_fun)
    img = decode_fun(encoded_bytes, mode=mode)
    assert img[None].is_contiguous(memory_format=torch.channels_last)

    pil_img = Image.open(filename).convert(pil_mode)
    from_pil = F.pil_to_tensor(pil_img)
    assert_equal(img, from_pil)
    img += 123  # make sure image buffer wasn't freed by underlying decoding lib


@pytest.mark.skipif(not IS_LINUX, reason=HEIC_AVIF_MESSAGE)
@pytest.mark.parametrize("decode_fun", (decode_avif,))
def test_decode_avif(decode_fun):
    encoded_bytes = read_file(next(get_images(FAKEDATA_DIR, ".avif")))
    img = decode_fun(encoded_bytes)
    assert img.shape == (3, 100, 100)
    assert img[None].is_contiguous(memory_format=torch.channels_last)
    img += 123  # make sure image buffer wasn't freed by underlying decoding lib


# Note: decode_image fails because some of these files have a (valid) signature
# we don't recognize. We should probably use libmagic....
@pytest.mark.skipif(not IS_LINUX, reason=HEIC_AVIF_MESSAGE)
@pytest.mark.parametrize("decode_fun", (decode_avif, decode_heic))
@pytest.mark.parametrize(
    "mode, pil_mode",
    (
        (ImageReadMode.RGB, "RGB"),
        (ImageReadMode.RGB_ALPHA, "RGBA"),
        (ImageReadMode.UNCHANGED, None),
    ),
)
@pytest.mark.parametrize(
    "filename", Path("/home/nicolashug/dev/libavif/tests/data/").glob("*.avif"), ids=lambda p: p.name
)
def test_decode_avif_heic_against_pil(decode_fun, mode, pil_mode, filename):
    if "reversed_dimg_order" in str(filename):
        # Pillow properly decodes this one, but we don't (order of parts of the
        # image is wrong). This is due to a bug that was recently fixed in
        # libavif. Hopefully this test will end up passing soon with a new
        # libavif version https://github.com/AOMediaCodec/libavif/issues/2311
        pytest.xfail()
    import pillow_avif  # noqa

    encoded_bytes = read_file(filename)
    try:
        img = decode_fun(encoded_bytes, mode=mode)
    except RuntimeError as e:
        if any(
            s in str(e)
            for s in (
                "BMFF parsing failed",
                "avifDecoderParse failed: ",
                "file contains more than one image",
                "no 'ispe' property",
                "'iref' has double references",
                "Invalid image grid",
                "decode_heif failed: Invalid input: No 'meta' box",
            )
        ):
            pytest.skip(reason="Expected failure, that's OK")
        else:
            raise e
    assert img[None].is_contiguous(memory_format=torch.channels_last)
    if mode == ImageReadMode.RGB:
        assert img.shape[0] == 3
    if mode == ImageReadMode.RGB_ALPHA:
        assert img.shape[0] == 4

    if img.dtype == torch.uint16:
        img = F.to_dtype(img, dtype=torch.uint8, scale=True)
    try:
        from_pil = F.pil_to_tensor(Image.open(filename).convert(pil_mode))
    except RuntimeError as e:
        if any(s in str(e) for s in ("Invalid image grid", "Failed to decode image: Not implemented")):
            pytest.skip(reason="PIL failure")
        else:
            raise e

    if True:
        from torchvision.utils import make_grid

        g = make_grid([img, from_pil])
        F.to_pil_image(g).save((f"/home/nicolashug/out_images/{filename.name}.{pil_mode}.png"))

    is_decode_heic = getattr(decode_fun, "__name__", getattr(decode_fun, "name", None)) == "decode_heic"
    if mode == ImageReadMode.RGB and not is_decode_heic:
        # We don't compare torchvision's AVIF against PIL for RGB because
        # results look pretty different on RGBA images (other images are fine).
        # The result on torchvision basically just plainly ignores the alpha
        # channel, resuting in transparent pixels looking dark. PIL seems to be
        # using a sort of k-nn thing (Take a look at the resuting images)
        return
    if filename.name == "sofa_grid1x5_420.avif" and is_decode_heic:
        return

    torch.testing.assert_close(img, from_pil, rtol=0, atol=3)


@pytest.mark.skipif(not IS_LINUX, reason=HEIC_AVIF_MESSAGE)
@pytest.mark.parametrize("decode_fun", (decode_heic,))
def test_decode_heic(decode_fun):
    encoded_bytes = read_file(next(get_images(FAKEDATA_DIR, ".heic")))
    img = decode_fun(encoded_bytes)
    assert img.shape == (3, 100, 100)
    assert img[None].is_contiguous(memory_format=torch.channels_last)
    img += 123  # make sure image buffer wasn't freed by underlying decoding lib


@pytest.mark.parametrize("input_type", ("Path", "str", "tensor"))
@pytest.mark.parametrize("scripted", (False, True))
def test_decode_image_path(input_type, scripted):
    # Check that decode_image can support not just tensors as input
    path = next(get_images(IMAGE_ROOT, ".jpg"))
    if input_type == "Path":
        input = Path(path)
    elif input_type == "str":
        input = path
    elif input_type == "tensor":
        input = read_file(path)
    else:
        raise ValueError("Oops")

    if scripted and input_type == "Path":
        pytest.xfail(reason="Can't pass a Path when scripting")

    decode_fun = torch.jit.script(decode_image) if scripted else decode_image
    decode_fun(input)


def test_mode_str():
    # Make sure decode_image supports string modes. We just test decode_image,
    # not all of the decoding functions, but they should all support that too.
    # Torchscript fails when passing strings, which is expected.
    path = next(get_images(IMAGE_ROOT, ".png"))
    assert decode_image(path, mode="RGB").shape[0] == 3
    assert decode_image(path, mode="rGb").shape[0] == 3
    assert decode_image(path, mode="GRAY").shape[0] == 1
    assert decode_image(path, mode="RGBA").shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__])
