import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple

from . import _utils as det_utils
from .anchor_utils import DBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
from .. import vgg

from .retinanet import RetinaNet, RetinaNetHead  # TODO: Refactor both to inherit properly


__all__ = ['SSD']


class SSDHead(RetinaNetHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        nn.Module.__init__(self)
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)


class SSDClassificationHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            self.cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSDRegressionHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        super().__init__()
        self.bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            self.bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSD(RetinaNet):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 size: int = 300, image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 aspect_ratios: Optional[List[List[int]]] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400):
        nn.Module.__init__(self)

        if aspect_ratios is None:
            aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(backbone.parameters()).device
        tmp_img = torch.empty((1, 3, size, size), device=device)
        tmp_sizes = [x.size() for x in backbone(tmp_img).values()]
        out_channels = [x[1] for x in tmp_sizes]
        feature_map_sizes = [x[2] for x in tmp_sizes]

        assert len(feature_map_sizes) == len(aspect_ratios)

        self.backbone = backbone

        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        num_anchors = [2 + 2 * len(r) for r in aspect_ratios]
        self.head = SSDHead(out_channels, num_anchors, num_classes)

        self.anchor_generator = DBoxGenerator(size, feature_map_sizes, aspect_ratios)

        self.proposal_matcher = det_utils.Matcher(
            iou_thresh,
            iou_thresh,
            allow_low_quality_matches=True,
        )

        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(size, size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     anchors: List[Tensor]) -> Dict[str, Tensor]:
        pass


class SSDFeatureExtractorVGG(nn.Module):
    # TODO: That's the SSD300 extractor. handle the SDD500 case as well. See page 11, footernote 5.
    def __init__(self, backbone: nn.Module):
        super().__init__()
        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.block1 = nn.Sequential(
            *backbone[:maxpool4_pos]  # until conv4_3
        )
        self.block2 = nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
            nn.ReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
            nn.ReLU(inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # L2 regularization + Rescaling of 1st block's feature map
        x = self.block1(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in (self.block2, self.block3, self.block4, self.block5, self.block6):
            x = block(x)
            output.append(x)

        return OrderedDict(((str(i), v) for i, v in enumerate(output)))


def _vgg_backbone(backbone_name: str, pretrained: bool, trainable_layers: int = 3):
    backbone = vgg.__dict__[backbone_name](pretrained=pretrained).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorVGG(backbone)


def ssd_vgg16(pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True,
              trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg_backbone("vgg16", pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = SSD(backbone, num_classes, **kwargs)
    if pretrained:
        pass  # TODO: load pre-trained COCO weights
    return model
