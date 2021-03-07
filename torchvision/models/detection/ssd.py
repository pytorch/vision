from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple

from .backbone_utils import _validate_trainable_layers
from .. import vgg


__all__ = ['SSD']


class SSDHead(nn.Module):
    # TODO: Similar to RetinaNetHead. Perhaps abstract and reuse for one-shot detectors.
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class SSDClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSDRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSD(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors=(4, 6, 6, 6, 4, 4)):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     anchors: List[Tensor]) -> Dict[str, Tensor]:
        pass

    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        pass

    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        pass


class MultiFeatureMap(nn.Module):

    def __init__(self, feature_maps: nn.ModuleList):
        super().__init__()
        self.feature_maps = feature_maps

    def forward(self, x):
        output = []
        for block in self.feature_maps:
            x = block(x)
            output.append(x)
        return output


def _vgg16_mfm_backbone(pretrained, trainable_layers=3):
    backbone = vgg.vgg16(pretrained=pretrained).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    # Patch ceil_mode for all maxpool layers of backbone to get the same outputs as Fig2 of SSD papers
    for layer in backbone:
        if isinstance(layer, nn.MaxPool2d):
            layer.ceil_mode = True

    # Multiple Feature map definition - page 4, Fig 2 of SSD paper
    def build_feature_map_block(layers, out_channels):
        block = nn.Sequential(*layers)
        block.out_channels = out_channels
        return block

    feature_maps = nn.ModuleList([
        # Conv4_3 map
        build_feature_map_block(
            backbone[:23],  # until conv4_3
            # TODO: add L2 nomarlization + scaling?
            512
        ),
        # FC7 map
        build_feature_map_block(
            (
                *backbone[23:-1],  # until conv5_3
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),  # modified maxpool5
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
                nn.ReLU(inplace=True)
            ),
            1024
        ),
        # Conv8_2 map
        build_feature_map_block(
            (
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),
            512,
        ),
        # Conv9_2 map
        build_feature_map_block(
            (
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),
            256,
        ),
        # Conv10_2 map
        build_feature_map_block(
            (
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
            256,
        ),
        # Conv11_2 map
        build_feature_map_block(
            (
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
            256,
        ),
    ])

    return MultiFeatureMap(feature_maps)


def ssd_vgg16(pretrained=False, progress=True,
              num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg16_mfm_backbone(pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = SSD(backbone, num_classes, **kwargs)
    if pretrained:
        pass  # TODO: load pre-trained COCO weights
    return model
