from ....models.detection.backbone_utils import misc_nn_ops, LastLevelMaxPool, BackboneWithFPN
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

    # COPY-PASTED CODE FROM torchvision.models.detection.backbone_utils.resnet_fpn_backbone
    # =====================================================================================
    assert 0 <= trainable_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    # =====================================================================================
