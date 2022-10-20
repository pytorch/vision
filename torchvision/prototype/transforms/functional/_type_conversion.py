import unittest.mock
from typing import Any, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch
from torchvision.io.video import read_video
from torchvision.prototype import features
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer
from torchvision.transforms import functional as _F, functional_tensor as _FT


@torch.jit.unused
def decode_image_with_pil(encoded_image: torch.Tensor) -> features.Image:
    image = torch.as_tensor(np.array(PIL.Image.open(ReadOnlyTensorBuffer(encoded_image)), copy=True))
    if image.ndim == 2:
        image = image.unsqueeze(2)
    return features.Image(image.permute(2, 0, 1))


@torch.jit.unused
def decode_video_with_av(encoded_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with unittest.mock.patch("torchvision.io.video.os.path.exists", return_value=True):
        return read_video(ReadOnlyTensorBuffer(encoded_video))  # type: ignore[arg-type]


@torch.jit.unused
def to_image_tensor(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> features.Image:
    if isinstance(image, np.ndarray):
        output = torch.from_numpy(image).permute((2, 0, 1)).contiguous()
    elif isinstance(image, PIL.Image.Image):
        output = pil_to_tensor(image)
    else:  # isinstance(inpt, torch.Tensor):
        output = image
    return features.Image(output)


to_image_pil = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor

# We changed the names to align them with the new naming scheme. Still, `to_pil_image` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
to_pil_image = to_image_pil


def _num_value_bits(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 8
    elif dtype == torch.int8:
        return 7
    elif dtype == torch.int16:
        return 15
    elif dtype == torch.int32:
        return 31
    elif dtype == torch.int64:
        return 63
    else:
        raise TypeError(f"Number of value bits is only defined for integer dtypes, but got {dtype}.")


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input img should be Tensor Image")

    if image.dtype == dtype:
        return image

    float_input = image.is_floating_point()
    if torch.jit.is_scripting():
        # TODO: remove this branch as soon as `dtype.is_floating_point` is supported by JIT
        float_output = torch.tensor(0, dtype=dtype).is_floating_point()
    else:
        float_output = dtype.is_floating_point

    if float_input:
        # float to float
        if float_output:
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and dtype == torch.int64
        ):
            raise RuntimeError(f"The conversion from {image.dtype} to {dtype} cannot be performed safely.")

        # For data in the range `[0.0, 1.0]`, just multiplying by the maximum value of the integer range and converting
        # to the integer dtype  is not sufficient. For example, `torch.rand(...).mul(255).to(torch.uint8)` will only
        # be `255` if the input is exactly `1.0`. See https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # for a detailed analysis.
        # To mitigate this, we could round before we convert to the integer dtype, but this is an extra operation.
        # Instead, we can also multiply by the maximum value plus something close to `1`. See
        # https://github.com/pytorch/vision/pull/2078#issuecomment-613524965 for details.
        eps = 1e-3
        max_value = float(_FT._max_value(dtype))
        # We need to scale first since the conversion would otherwise turn the input range `[0.0, 1.0]` into the
        # discrete set `{0, 1}`.
        return image.mul(max_value + 1.0 - eps).to(dtype)
    else:
        # int to float
        if float_output:
            return image.to(dtype).div_(_FT._max_value(image.dtype))

        # int to int
        num_value_bits_input = _num_value_bits(image.dtype)
        num_value_bits_output = _num_value_bits(dtype)

        if num_value_bits_input > num_value_bits_output:
            return image.bitwise_right_shift(num_value_bits_input - num_value_bits_output).to(dtype)
        else:
            # The bitshift kernel is not vectorized
            #  https://github.com/pytorch/pytorch/blob/703c19008df4700b6a522b0ae5c4b6d5ffc0906f/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L315-L322
            #  This results in the multiplication actually being faster.
            # TODO: If the bitshift kernel is optimized in core, replace the computation below with
            #  `image.to(dtype).bitwise_left_shift_(num_value_bits_output - num_value_bits_input)`
            max_value_input = float(_FT._max_value(dtype))
            max_value_output = float(_FT._max_value(image.dtype))
            factor = int((max_value_input + 1) // (max_value_output + 1))
            return image.to(dtype).mul_(factor)
