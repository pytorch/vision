import torch


def _new_empty_tensor(x, shape):
    """
    """
    return torch.ops.torchvision._new_empty_tensor_op(x, shape)
