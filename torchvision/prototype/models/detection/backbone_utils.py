from typing import Callable, Optional, List

from torch import nn

from ....models.detection.backbone_utils import misc_nn_ops, _resnet_backbone_config, BackboneWithFPN, ExtraFPNBlock
from .. import resnet
from .._api import Weights


def resnet_fpn_backbone(
    backbone_name: str,
    weights: Optional[Weights],
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:

    backbone = resnet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _resnet_backbone_config(backbone, trainable_layers, returned_layers, extra_blocks)
