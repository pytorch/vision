import torch.nn as nn

from ._utils import Conv3DSimple, Conv3DNoTemporal
from .video_stems import get_default_stem
from .video_trunk import VideoTrunkBuilder, BasicBlock, Bottleneck


__all__ = ["mcX", "rmcX"]


def mcX(model_depth, X=3, use_pool1=False, **kwargs):
    """Generate mixed convolution network as in
        https://arxiv.org/abs/1711.11248

    Args:
        model_depth (int): trunk depth - supports most resnet depths
        X (int): Up to which layers are convolutions 3D
        use_pool1 (bool, optional): Add pooling layer to the stem. Defaults to False.

    Returns:
        nn.Module: mcX video trunk
    """
    assert X > 1 and X <= 5
    conv_makers = [Conv3DSimple] * (X - 2)
    while len(conv_makers) < 5:
        conv_makers.append(Conv3DNoTemporal)

    if model_depth < 50:
        block = BasicBlock
    else:
        block = Bottleneck

    model = VideoTrunkBuilder(block=block, conv_makers=conv_makers, model_depth=model_depth,
                              stem=get_default_stem(use_pool1=use_pool1), **kwargs)

    return model


def rmcX(model_depth, X=3, use_pool1=False, **kwargs):
    """Generate reverse mixed convolution network as in
        https://arxiv.org/abs/1711.11248

    Args:
        model_depth (int): trunk depth - supports most resnet depths
        X (int): Up to which layers are convolutions 2D
        use_pool1 (bool, optional): Add pooling layer to the stem. Defaults to False.

    Returns:
        nn.Module: mcX video trunk
    """
    assert X > 1 and X <= 5

    conv_makers = [Conv3DNoTemporal] * (X - 2)
    while len(conv_makers) < 5:
        conv_makers.append(Conv3DSimple)

    if model_depth < 50:
        block = BasicBlock
    else:
        block = Bottleneck

    model = VideoTrunkBuilder(block=block, conv_makers=conv_makers, model_depth=model_depth,
                              stem=get_default_stem(use_pool1=use_pool1), **kwargs)

    return model
