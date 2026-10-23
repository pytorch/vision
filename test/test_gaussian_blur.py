import numpy as np
import PIL.Image
import pytest
import torch
import torch.nn.functional as nnf
from common_utils import cpu_and_cuda
from torchvision import tv_tensors
from torchvision.transforms import _functional_tensor as F_t, functional as F
from torchvision.transforms.v2 import functional as F_v2


def dense_gaussian_blur(image, kernel_size, sigma):
    # Use a double precision, two-dimensional convolution as an independent
    # reference, including reflection at the corners of the image.
    kernels = []
    for size, std in zip(kernel_size, sigma):
        x = torch.arange(size, dtype=torch.float64, device=image.device) - (size - 1) / 2
        kernel = (-0.5 * (x / std).square()).exp()
        kernels.append(kernel / kernel.sum())
    kernel = kernels[1][:, None] * kernels[0][None, :]
    channels = image.shape[-3]
    padded = nnf.pad(
        image.double().reshape(-1, *image.shape[-3:]),
        [kernel_size[0] // 2] * 2 + [kernel_size[1] // 2] * 2,
        mode="reflect",
    )
    return nnf.conv2d(padded, kernel.expand(channels, 1, *kernel.shape), groups=channels).reshape(image.shape)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.uint8])
@pytest.mark.parametrize("kernel_size", [[31, 41], [41, 31], [31, 33], [31, 35], [33, 33], [1, 65], [65, 1]])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "noncontiguous"])
def test_gaussian_blur_large_kernel(fn, device, dtype, kernel_size, layout):
    # Both spatial dimensions must exceed the reflection padding.
    image = torch.randint(0, 256, (2, 3, 48, 72), device=device).to(dtype)
    if layout == "channels_last":
        image = image.contiguous(memory_format=torch.channels_last)
    elif layout == "noncontiguous":
        image = image.transpose(-1, -2)
    sigma = [5.0, 9.0]
    original = image.clone()
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    actual = fn(image, kernel_size, sigma)
    assert actual.dtype == dtype
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    if dtype == torch.uint8:
        # Round only after both passes. Half-integer ties can round differently
        # because the summation order differs from the dense convolution.
        torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.501)
    elif dtype == torch.float64:
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    else:
        # The dense fallback also accumulates up to 1024 float32 products.
        torch.testing.assert_close(actual.double(), expected, rtol=5e-6, atol=1e-4)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
def test_gaussian_blur_large_kernel_backward(fn):
    image = torch.rand(1, 1, 22, 18, dtype=torch.float64, requires_grad=True)
    kernel_size, sigma = [31, 41], [7.0, 3.0]
    actual = fn(image, kernel_size, sigma)
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, image, weights)[0]
    expected_grad = torch.autograd.grad(expected, image, weights)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda x: fn(x, kernel_size, sigma), (image,), fast_mode=True)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
def test_gaussian_blur_large_kernel_workspace(fn):
    # CPU float64 convolution uses an im2col workspace. A full 31x41 kernel
    # allocates about 50 MiB for this small image (see #7413).
    image = torch.rand(1, 64, 80, dtype=torch.float64)
    kernel_size = [31, 41]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, acc_events=True
    ) as prof:
        fn(image, kernel_size, [5.0, 7.0])
    largest_allocation = max(event.self_cpu_memory_usage for event in prof.events())
    # Allow workspace linear in the longest kernel dimension, rather than the
    # product of both dimensions. The factor of two allows for padding.
    assert largest_allocation <= 2 * image.numel() * image.element_size() * max(kernel_size)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
@pytest.mark.parametrize("sigma", [[0.1, 0.2], [1000.0, 500.0]])
def test_gaussian_blur_large_kernel_scripted(fn, sigma):
    image = torch.rand(1, 24, 32, dtype=torch.float64)
    kernel_size = [31, 41]
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    torch.testing.assert_close(torch.jit.script(fn)(image, kernel_size, sigma), expected)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gaussian_blur_large_kernel_autocast(fn, device, dtype):
    image = torch.rand(1, 3, 24, 32, device=device)
    kernel_size, sigma = [31, 41], [5.0, 7.0]
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    with torch.autocast(device_type=device, dtype=dtype):
        actual = fn(image, kernel_size, sigma)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.006)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gaussian_blur_large_kernel_low_precision(fn, device, dtype):
    image = torch.rand(1, 3, 24, 32, device=device).to(dtype)
    kernel_size, sigma = [31, 41], [5.0, 7.0]
    actual = fn(image, kernel_size, sigma)
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    assert actual.dtype == image.dtype
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.006)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
def test_gaussian_blur_large_kernel_default_sigma(fn):
    image = torch.rand(3, 24, 32, dtype=torch.float64)
    kernel_size = [31, 41]
    sigma = [size * 0.15 + 0.35 for size in kernel_size]
    torch.testing.assert_close(fn(image, kernel_size), dense_gaussian_blur(image, kernel_size, sigma))


@pytest.mark.parametrize("shape", [(2, 3, 1, 24, 32), (0, 3, 24, 32), (2, 0, 24, 32)])
def test_gaussian_blur_large_kernel_video(shape):
    image = tv_tensors.Video(torch.rand(shape, dtype=torch.float64))
    actual = F_v2.gaussian_blur(image, [31, 41], [5.0, 7.0])
    assert actual.shape == image.shape
    assert isinstance(actual, tv_tensors.Video)
    if image.numel():
        torch.testing.assert_close(actual, dense_gaussian_blur(image, [31, 41], [5.0, 7.0]))


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
def test_gaussian_blur_large_kernel_pil(fn):
    image = torch.randint(0, 256, (3, 24, 32), dtype=torch.uint8)
    pil_image = PIL.Image.fromarray(image.permute(1, 2, 0).numpy())
    output = fn(pil_image, [31, 41], [5.0, 7.0])
    actual = torch.from_numpy(np.array(output)).permute(2, 0, 1)
    expected = dense_gaussian_blur(image, [31, 41], [5.0, 7.0])
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.501)


@pytest.mark.parametrize("fn", [F.gaussian_blur, F_v2.gaussian_blur])
@pytest.mark.parametrize("center_weight", [0.49, 0.5, 0.51])
def test_gaussian_blur_large_kernel_rounding(fn, center_weight):
    # Choose sigma so that blurring a one-unit impulse produces a value close
    # to a half-integer. This catches rounding the intermediate horizontal pass.
    x = np.arange(-16, 17, dtype=np.float64)
    low, high = 0.1, 2.0
    for _ in range(50):
        sigma_x = (low + high) / 2
        weight = 1 / np.exp(-0.5 * (x / sigma_x) ** 2).sum()
        if weight > center_weight**0.5:
            low = sigma_x
        else:
            high = sigma_x
    image = torch.full((1, 35, 35), 100, dtype=torch.uint8)
    image[0, 17, 17] = 101
    # A batch large enough to exercise separable CPU dispatch.
    actual = fn(image.unsqueeze(0).expand(256, -1, -1, -1), [33, 33], [sigma_x, sigma_x])[0]
    expected = dense_gaussian_blur(image, [33, 33], [sigma_x, sigma_x])
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.501)
    if center_weight != 0.5:
        assert actual[0, 17, 17].item() == round(100 + center_weight)


@pytest.mark.parametrize("above_threshold", [False, True])
def test_gaussian_blur_cpu_work_threshold(above_threshold):
    numel = (2**28) // (33 * 33) + int(above_threshold)
    image = torch.empty(numel)
    assert F_t._should_use_separable_gaussian_blur(image, [33, 33]) == above_threshold


@pytest.mark.parametrize("device", cpu_and_cuda())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_blur_dispatch(device, dtype):
    image = torch.empty(1, 64, 80, device=device, dtype=dtype)
    assert F_t._should_use_separable_gaussian_blur(image, [31, 41]) == (device == "cuda" or dtype == torch.float64)
    assert not F_t._should_use_separable_gaussian_blur(image, [1, 2049])
    assert not F_t._should_use_separable_gaussian_blur(image, [31, 33])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_separable_gaussian_blur_cpu(dtype):
    # Exercise the two-pass kernel even when dispatch keeps small CPU inputs
    # on the dense path to avoid the overhead of a second convolution.
    image = torch.rand(2, 3, 24, 32).to(dtype)
    kernel_size, sigma = [31, 41], [5.0, 7.0]
    kernel_x = F_t._get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, image.device)
    kernel_y = F_t._get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, image.device)
    actual = F_t._separable_gaussian_blur(image, kernel_x, kernel_y)
    expected = dense_gaussian_blur(image, kernel_size, sigma)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0.006 if dtype != torch.float32 else 1e-6)
