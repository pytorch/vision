# TODO: Implement v1 and v2 versions of the mobile ViT model.

from torch import nn
from torchvision.utils import _log_api_usage_once

__all__ = ["MobileViT"]

# TODO: Update this...
# Paper links: v1 https://arxiv.org/abs/2110.02178
# v2
class MobileViT(nn.Module):
    """
    Implements MobileViT from the `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_ paper.
    Args:
        TODO: Arguments to be updated...
    """

    def __init__(
        self,
    ):
        super().__init__()
        _log_api_usage_once(self)
        # TODO: Add blocks...

    # TODO: This is the core thing to implement...
    def forward(self, x):
        return x
