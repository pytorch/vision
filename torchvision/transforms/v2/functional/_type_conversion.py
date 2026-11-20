from types import SimpleNamespace
from typing import Union

import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import functional as _F


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
def to_cvcuda_tensor(inpt: torch.Tensor, *, device=None):
    """Convert a CHW/NCHW Torch tensor to CV-CUDA without scaling or permuting values.

    CPU inputs move to the current CUDA device; CUDA inputs keep their device unless
    ``device`` is specified. Non-contiguous inputs are copied. Otherwise storage is shared.
    """
    if not isinstance(inpt, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(inpt)}.")
    if inpt.ndim not in (3, 4):
        raise ValueError("Input must have CHW or NCHW shape (rank 3 or 4).")
    if inpt.requires_grad:
        raise ValueError("CV-CUDA does not support autograd. Detach the input explicitly.")
    from ._cvcuda import _import_cvcuda

    cvcuda = _import_cvcuda()
    device = torch.device(device) if device is not None else (inpt.device if inpt.is_cuda else torch.device("cuda"))
    if device.type != "cuda":
        raise ValueError("CV-CUDA tensors require a CUDA device.")
    inpt = inpt.to(device=device).contiguous()
    interface = dict(inpt.__cuda_array_interface__)
    interface.update(version=3, stream=torch.cuda.current_stream(inpt.device).cuda_stream or -1)
    owner = SimpleNamespace(_tensor=inpt, __cuda_array_interface__=interface)
    with torch.cuda.device(inpt.device):
        return cvcuda.as_tensor(owner, "CHW" if inpt.ndim == 3 else "NCHW")


@torch.jit.unused
def cvcuda_to_tensor(inpt) -> torch.Tensor:
    """Return a zero-copy CUDA Torch tensor, preserving shape, dtype, and device."""
    from ._cvcuda import _import_cvcuda

    cvcuda = _import_cvcuda()
    if not isinstance(inpt, cvcuda.Tensor):
        raise TypeError(f"Input must be a cvcuda.Tensor, got {type(inpt)}.")
    buffer = inpt.cuda()
    # ExternalBuffer exports storage through CAI; DLPack only supplies its device here.
    device = torch.device("cuda", buffer.__dlpack_device__()[1])
    with torch.cuda.device(device):
        return torch.as_tensor(buffer, device=device)
