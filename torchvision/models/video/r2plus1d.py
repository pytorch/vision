import torch.nn as nn

from ._utils import Conv2Plus1D
from .video_stems import get_r2plus1d_stem
from .video_trunk import VideoTrunkBuilder, BasicBlock, Bottleneck


__all__ = ["r2plus1d"]


def r2plus1d(model_depth, use_pool1=False, **kwargs):
    """Constructor for R(2+1)D network as described in
    https://arxiv.org/abs/1711.11248

    Args:
        model_depth (int): Depth of the model - standard resnet depths apply
        use_pool1 (bool, optional): Should we use the pooling layer? Defaults to False
    Returns:
        nn.Module: An R(2+1)D video backbone
    """
    convs = [Conv2Plus1D()] * 4
    if model_depth < 50:
        block = BasicBlock
    else:
        block = Bottleneck

    model = VideoTrunkBuilder(
        block=block, conv_makers=convs, model_depth=model_depth,
        stem=get_r2plus1d_stem(use_pool1), **kwargs)
    return model
