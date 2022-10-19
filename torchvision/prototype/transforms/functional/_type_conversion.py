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
        max_val = float(_FT._max_value(dtype))
        # We need to scale first since the conversion would otherwise turn the input range `[0.0, 1.0]` into the
        # discrete set `{0, 1}`.
        return image.mul(max_val + 1.0 - eps).to(dtype)
    else:
        max_input_val = float(_FT._max_value(image.dtype))

        # int to float
        if float_output:
            return image.to(dtype).div_(max_input_val)

        # int to int
        # TODO: The `factor`'s below are by definition powers of 2. Instead of multiplying and dividing the inputs to
        #  get to the desired value range, we can probably speed this up significantly with bitshifts. However, we
        #  probably need to be careful when converting from signed to unsigned dtypes and vice versa.
        max_output_val = float(_FT._max_value(dtype))

        if max_input_val > max_output_val:
            # We technically don't need to convert to `int` here, but it speeds the division
            factor = int((max_input_val + 1) / (max_output_val + 1))
            # We need to scale first since the output dtype cannot hold all values in the input range
            return image.div(factor, rounding_mode="floor").to(dtype)
        else:
            # We need to convert to `int` or otherwise the multiplication will turn the image into floating point. Or,
            # to be more exact, the inplace multiplication will fail.
            factor = int((max_output_val + 1) / (max_input_val + 1))
            return image.to(dtype).mul_(factor)
