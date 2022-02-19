import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..utils import _log_api_usage_once


class DropBlock2d(nn.Module):
    """
    Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.

    Args:
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``
    """

    def __init__(self, p: float, block_size: int, inplace: bool = False) -> None:
        super(DropBlock2d, self).__init__()
        _log_api_usage_once(self)

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(
            input=mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size
        )
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s
