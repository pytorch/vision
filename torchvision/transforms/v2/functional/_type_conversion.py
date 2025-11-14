from typing import TYPE_CHECKING, Union

import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import functional as _F
from torchvision.utils import _log_api_usage_once

from ._utils import _import_cvcuda

if TYPE_CHECKING:
    import cvcuda  # type: ignore[import-not-found]


@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PIL.Image.Image):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(inpt)} instead."
        )
    return tv_tensors.Image(output)


to_pil_image = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor


@torch.jit.unused
def to_cvcuda_tensor(inpt: torch.Tensor) -> "cvcuda.Tensor":
    """See :class:``~torchvision.transforms.v2.ToCVCUDATensor`` for details."""
    cvcuda = _import_cvcuda()
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(to_cvcuda_tensor)
    if not isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
        raise TypeError(f"inpt should be ``torch.Tensor`` or ``tv_tensors.Image``. Got {type(inpt)}.")
    if inpt.ndim != 4:
        raise ValueError(f"pic should be 4 dimensional. Got {inpt.ndim} dimensions.")
    # Convert to NHWC as CVCUDA transforms do not support NCHW
    inpt = inpt.permute(0, 2, 3, 1)
    return cvcuda.as_tensor(inpt.cuda().contiguous(), cvcuda.TensorLayout.NHWC)


@torch.jit.unused
def cvcuda_to_tensor(cvcuda_img: "cvcuda.Tensor") -> torch.Tensor:
    """See :class:``~torchvision.transforms.v2.CVCUDAToTensor`` for details."""
    cvcuda = _import_cvcuda()
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(cvcuda_to_tensor)
    if not isinstance(cvcuda_img, cvcuda.Tensor):
        raise TypeError(f"cvcuda_img should be ``cvcuda.Tensor``. Got {type(cvcuda_img)}.")
    cuda_tensor = torch.as_tensor(cvcuda_img.cuda(), device="cuda")
    if cvcuda_img.ndim != 4:
        raise ValueError(f"Image should be 4 dimensional. Got {cuda_tensor.ndim} dimensions.")
    # Convert to NCHW shape from CVCUDA default NHWC
    img = cuda_tensor.permute(0, 3, 1, 2)
    return img
