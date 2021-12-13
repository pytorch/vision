from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Type, Union

from torch import nn
from torchvision.prototype.transforms import Kinect400Eval
from torchvision.transforms.functional import InterpolationMode

from ....models.video.resnet import (
    BasicBlock,
    BasicStem,
    Bottleneck,
    Conv2Plus1D,
    Conv3DSimple,
    Conv3DNoTemporal,
    R2Plus1dStem,
    VideoResNet,
)
from .._api import WeightsEnum, Weights
from .._meta import _KINETICS400_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "VideoResNet",
    "R3D_18_Weights",
    "MC3_18_Weights",
    "R2Plus1D_18_Weights",
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
]


def _video_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
    layers: List[int],
    stem: Callable[..., nn.Module],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VideoResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = VideoResNet(block, conv_makers, layers, stem, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "size": (112, 112),
    "categories": _KINETICS400_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/video_classification",
}


class R3D_18_Weights(WeightsEnum):
    Kinetics400_V1 = Weights(
        url="https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
        transforms=partial(Kinect400Eval, crop_size=(112, 112), resize_size=(128, 171)),
        meta={
            **_COMMON_META,
            "acc@1": 52.75,
            "acc@5": 75.45,
        },
    )
    default = Kinetics400_V1


class MC3_18_Weights(WeightsEnum):
    Kinetics400_V1 = Weights(
        url="https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
        transforms=partial(Kinect400Eval, crop_size=(112, 112), resize_size=(128, 171)),
        meta={
            **_COMMON_META,
            "acc@1": 53.90,
            "acc@5": 76.29,
        },
    )
    default = Kinetics400_V1


class R2Plus1D_18_Weights(WeightsEnum):
    Kinetics400_V1 = Weights(
        url="https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
        transforms=partial(Kinect400Eval, crop_size=(112, 112), resize_size=(128, 171)),
        meta={
            **_COMMON_META,
            "acc@1": 57.50,
            "acc@5": 78.81,
        },
    )
    default = Kinetics400_V1


@handle_legacy_interface(weights=("pretrained", R3D_18_Weights.Kinetics400_V1))
def r3d_18(*, weights: Optional[R3D_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    weights = R3D_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv3DSimple] * 4,
        [2, 2, 2, 2],
        BasicStem,
        weights,
        progress,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", MC3_18_Weights.Kinetics400_V1))
def mc3_18(*, weights: Optional[MC3_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    weights = MC3_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv3DSimple] + [Conv3DNoTemporal] * 3,  # type: ignore[list-item]
        [2, 2, 2, 2],
        BasicStem,
        weights,
        progress,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", R2Plus1D_18_Weights.Kinetics400_V1))
def r2plus1d_18(*, weights: Optional[R2Plus1D_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    weights = R2Plus1D_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv2Plus1D] * 4,
        [2, 2, 2, 2],
        R2Plus1dStem,
        weights,
        progress,
        **kwargs,
    )
