from ....models.detection.backbone_utils import misc_nn_ops, _resnet_backbone_config
from .. import resnet


def resnet_fpn_backbone(
    backbone_name,
    weights,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None,
):
    backbone = resnet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _resnet_backbone_config(backbone, trainable_layers, returned_layers, extra_blocks)
