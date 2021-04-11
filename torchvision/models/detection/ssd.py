import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional

from . import _utils as det_utils
from .anchor_utils import DBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
from .. import vgg
from ..utils import load_state_dict_from_url

from .retinanet import RetinaNet, RetinaNetHead, RetinaNetRegressionHead, _sum  # TODO: Refactor to inherit properly


__all__ = ['SSD', 'ssd300_vgg16']

model_urls = {
    'ssd300_vgg16_coco': None,  # TODO: Add url with weights
}


def _xavier_init(conv: nn.Module):
    for layer in conv.children():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)


class SSDHead(RetinaNetHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, positive_fraction: float,
                 box_coder: det_utils.BoxCoder):
        nn.Module.__init__(self)
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes, positive_fraction)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors, box_coder)


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        nn.Module.__init__(self)
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.module_list:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, positive_fraction: float):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        losses = []

        cls_logits = head_outputs['cls_logits']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum().item()

            gt_classes_target = torch.zeros((cls_logits_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = \
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            classification_loss = F.cross_entropy(cls_logits_per_image, gt_classes_target,
                                                  reduce=False)

            # Hard Negative Sampling
            background_idxs_per_image = torch.logical_not(foreground_idxs_per_image)
            num_background = matched_idxs_per_image.size(0) - num_foreground
            num_negative = min(num_background, int(self.neg_to_pos_ratio * num_foreground))

            foreground_loss = classification_loss[foreground_idxs_per_image]
            background_loss = classification_loss[background_idxs_per_image].sort(descending=True)[0][:num_negative]

            losses.append((foreground_loss.sum() + background_loss.sum()) / max(1, num_foreground))

        return _sum(losses) / len(targets)


class SSDRegressionHead(SSDScoringHead, RetinaNetRegressionHead):  # TODO: Refactor to avoid multiple inheritance
    def __init__(self, in_channels: List[int], num_anchors: List[int], box_coder: det_utils.BoxCoder):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        SSDScoringHead.__init__(self, bbox_reg, 4)
        self.box_coder = box_coder
        self._l1_loss = torch.nn.functional.smooth_l1_loss  # TODO: Discuss/refactor this workaround


class SSDFeatureExtractor(nn.Module):
    def __init__(self, aspect_ratios: List[List[int]]):
        super().__init__()
        self.aspect_ratios = aspect_ratios


class SSD(RetinaNet):
    def __init__(self, backbone: SSDFeatureExtractor, size: int, num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25):
        nn.Module.__init__(self)

        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(backbone.parameters()).device
        tmp_img = torch.empty((1, 3, size, size), device=device)
        tmp_sizes = [x.size() for x in backbone(tmp_img).values()]
        out_channels = [x[1] for x in tmp_sizes]
        feature_map_sizes = [x[2] for x in tmp_sizes]

        assert len(feature_map_sizes) == len(backbone.aspect_ratios)

        self.backbone = backbone

        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))

        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        self.num_anchors = [2 + 2 * len(r) for r in backbone.aspect_ratios]
        self.head = SSDHead(out_channels, self.num_anchors, num_classes, positive_fraction, self.box_coder)

        self.anchor_generator = DBoxGenerator(size, feature_map_sizes, backbone.aspect_ratios)

        self.proposal_matcher = det_utils.Matcher(iou_thresh, iou_thresh)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(size, size, image_mean, image_std,
                                                  size_divisible=1)  # TODO: Discuss/refactor this workaround

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    def _anchors_per_level(self, features: List[Tensor], HWA: int):
        # TODO: Discuss/refactor this workaround
        num_anchors_per_level = [x.size(2) * x.size(3) * anchors for x, anchors in zip(features, self.num_anchors)]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        A = HWA // HW
        return [hw * A for hw in num_anchors_per_level]


class SSDFeatureExtractorVGG(SSDFeatureExtractor):
    def __init__(self, backbone: nn.Module, extra: nn.ModuleList, aspect_ratios: List[List[int]]):
        super().__init__(aspect_ratios)
        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(
            *backbone[:maxpool4_pos]  # until conv4_3
        )
        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            fc,
        ))
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # L2 regularization + Rescaling of 1st block's feature map
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _vgg_extractor(backbone_name: str, highres: bool, pretrained: bool, trainable_layers: int = 3):
    backbone = vgg.__dict__[backbone_name](pretrained=pretrained).features
    # SDD300 case - page 4, Fig 2 of SSD paper
    extra = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
            nn.ReLU(inplace=True),
        )
    ])
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    if highres:
        # Additional layers for the SDD512 case. See page 11, footernote 5.
        extra.append(nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
            nn.ReLU(inplace=True),
        ))
        aspect_ratios.append([2])
    _xavier_init(extra)

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorVGG(backbone, extra, aspect_ratios)


def ssd300_vgg16(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg_extractor("vgg16", False, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = SSD(backbone, 300, num_classes, **kwargs)
    if pretrained:
        weights_name = 'ssd300_vgg16_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model
