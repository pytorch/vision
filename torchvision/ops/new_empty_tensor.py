import torch
from torch.jit.annotations import List
from torch import Tensor


def _new_empty_tensor(x: Tensor, shape: List[int]) -> Tensor:
    """
    Arguments:
        input (Tensor): input tensor
        shape List[int]: the new empty tensor shape

    Returns:
        output (Tensor)
    """
    return torch.ops.torchvision._new_empty_tensor_op(x, shape)
