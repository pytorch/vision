import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .poolers import Pooler
from .roi_heads import RoIHeads

from .._utils import IntermediateLayerGetter
from . import fpn as fpn_module

from torchvision.ops import misc as misc_nn_ops

from maskrcnn_benchmark.layers import FrozenBatchNorm2d


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

def load_resnet_c2_format(f):
    from maskrcnn_benchmark.utils.c2_model_loading import _load_c2_pickled_weights, _C2_STAGE_NAMES, _rename_weights_for_resnet
    state_dict = _load_c2_pickled_weights(f)
    conv_body = "R-50-FPN"
    arch = conv_body.replace("-C4", "").replace("-C5", "").replace("-FPN", "")
    arch = arch.replace("-RETINANET", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    return state_dict

def build_resnet_fpn_backbone(backbone_name):
    from .. import resnet
    from .._utils import IntermediateLayerGetter
    from . import fpn as fpn_module

    backbone = resnet.__dict__[backbone_name](
        # pretrained=False,
        pretrained=True,
        norm_layer=FrozenBatchNorm2d)

    # del backbone.avgpool
    # del backbone.fc

    if False:
        state_dict = load_resnet_c2_format('/private/home/fmassa/.torch/models/R-50.pkl')
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict
        load_state_dict(backbone, state_dict)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    for name, parameter in body.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    in_channels_stage2 = 256  # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = 256  # cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

def build_backbone_with_fpn(backbone, return_layers, in_channels_list, out_channels):
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

def build_resnet_fpn_backbone_(backbone_name):
    from .. import resnet
    backbone = resnet.__dict__[backbone_name](
        # pretrained=False,
        pretrained=True,
        norm_layer=FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = 256
    out_channels = 256
    in_channels_list=[
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    return build_backbone_with_fpn(backbone, return_layers, in_channels_list, out_channels)


class MaskRCNN(GeneralizedRCNN):
    def __init__(self,
                 backbone, num_classes,
                 #
                 rpn_anchor_sizes=None, rpn_aspect_ratios=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 #
                 box_resolution=7, box_scales=None, box_sampling_ratio=2,
                 representation_size=1024,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 #
                 mask_resolution=14, mask_scales=None, mask_sampling_ratio=2,
                 mask_layers=None, mask_dilation=1,
                 mask_discretization_size=28):
        out_channels = backbone.out_channels
        rpn = build_rpn(
            out_channels,
            rpn_anchor_sizes, rpn_aspect_ratios,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
        )
        roi_heads = build_roi_heads(
            out_channels, num_classes,
            box_resolution, box_scales, box_sampling_ratio,
            representation_size,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            #
            True,
            mask_resolution, mask_scales, mask_sampling_ratio,
            mask_layers, mask_dilation,
            mask_discretization_size
        )

        super(MaskRCNN, self).__init__(backbone, rpn, roi_heads)



def maskrcnn_resnet50_fpn(pretrained=False, num_classes=81, **kwargs):
    backbone = build_resnet_fpn_backbone('resnet50')
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        pass
    return model


def build_rpn(in_channels,
              anchor_sizes=None, aspect_ratios=None,
              pre_nms_top_n_train=2000, pre_nms_top_n_test=1000,
              post_nms_top_n_train=2000, post_nms_top_n_test=1000,
              nms_thresh=0.7,
              fg_iou_thresh=0.7, bg_iou_thresh=0.3,
              batch_size_per_image=256, positive_fraction=0.5):

    if anchor_sizes is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    head = RPNHead(
        in_channels, anchor_generator.num_anchors_per_location()[0]
    )

    pre_nms_top_n = dict(training=pre_nms_top_n_train, testing=pre_nms_top_n_test)
    post_nms_top_n = dict(training=post_nms_top_n_train, testing=post_nms_top_n_test)

    return RegionProposalNetwork(anchor_generator, head,
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            pre_nms_top_n, post_nms_top_n, nms_thresh)


class TwoMLPHead(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            conv = misc_nn_ops.Conv2d(next_feature, layer_features, kernel_size=3,
                    stride=1, padding=dilation, dilation=dilation)
            d["mask_fcn{}".format(layer_idx)] = conv
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNC4Predictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNC4Predictor, self).__init__(OrderedDict([
            ("conv5_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


def build_roi_heads(in_channels, num_classes,
                    resolution=7, scales=None, sampling_ratio=2,
                    representation_size=1024,
                    score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
                    fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                    batch_size_per_image=512, positive_fraction=0.25,
                    bbox_reg_weights=None,
                    #
                    mask_on=True,
                    mask_resolution=14, mask_scales=None, mask_sampling_ratio=2,
                    mask_layers=None, mask_dilation=1,
                    mask_discretization_size=28
                    ):

    if scales is None:
        scales = (0.25, 0.125, 0.0625, 0.03125)

    if bbox_reg_weights is None:
        bbox_reg_weights = (10., 10., 5., 5.)

    if mask_scales is None:
        mask_scales = scales

    if mask_layers is None:
        mask_layers = (256, 256, 256, 256)

    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    feature_extractor = TwoMLPHead(
            in_channels * resolution ** 2,
            representation_size)
    box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

    mask_pooler = None
    mask_head = None
    mask_predictor = None
    if mask_on:
        mask_pooler = Pooler(
            output_size=(mask_resolution, mask_resolution),
            scales=mask_scales,
            sampling_ratio=mask_sampling_ratio,
        )

        mask_head = MaskRCNNHeads(in_channels, mask_layers, mask_dilation)
        mask_dim_reduced = mask_layers[-1]
        mask_predictor = MaskRCNNC4Predictor(in_channels, mask_dim_reduced, num_classes)

    return RoIHeads(pooler, feature_extractor, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Mask
            mask_pooler,
            mask_head,
            mask_predictor,
            mask_discretization_size
            )
