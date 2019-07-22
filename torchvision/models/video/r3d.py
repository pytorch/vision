import torch.nn as nn

from ._utils import Conv3DSimple
from .video_stems import get_default_stem
from .video_trunk import VideoTrunkBuilder, BasicBlock, Bottleneck

__all__ = ["r3d"]


def r3d(model_depth, use_pool1=False, **kwargs):
    """Constructor of a r3d network as in
    https://arxiv.org/abs/1711.11248

    Args:
        model_depth (int): resnet trunk depth
        use_pool1 (bool, optional): Add pooling layer to the stem. Defaults to False

    Returns:
        nn.Module: R3D network trunk
    """

    conv_makers = [Conv3DSimple()] * 4
    if model_depth < 50:
        block = BasicBlock
    else:
        block = Bottleneck

    model = VideoTrunkBuilder(block=block, conv_makers=conv_makers, model_depth=model_depth,
                              stem=get_default_stem(use_pool1=use_pool1), **kwargs)
    return model
