import copy
from collections import OrderedDict

import pytest
import torch
from common_utils import assert_equal
from torchvision.models.detection import _utils, backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class TestModelsDetectionUtils:
    def test_balanced_positive_negative_sampler(self):
        sampler = _utils.BalancedPositiveNegativeSampler(4, 0.25)
        # keep all 6 negatives first, then add 3 positives, last two are ignore
        matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
        pos, neg = sampler(matched_idxs)
        # we know the number of elements that should be sampled for the positive (1)
        # and the negative (3), and their location. Let's make sure that they are
        # there
        assert pos[0].sum() == 1
        assert pos[0][6:9].sum() == 1
        assert neg[0].sum() == 3
        assert neg[0][0:6].sum() == 3

    def test_box_linear_coder(self):
        box_coder = _utils.BoxLinearCoder(normalize_by_size=True)
        # Generate a random 10x4 boxes tensor, with coordinates < 50.
        boxes = torch.rand(10, 4) * 50
        boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression
        boxes[:, 2:] += boxes[:, :2]

        proposals = torch.tensor([0, 0, 101, 101] * 10).reshape(10, 4).float()

        rel_codes = box_coder.encode(boxes, proposals)
        pred_boxes = box_coder.decode(rel_codes, boxes)
        torch.allclose(proposals, pred_boxes)

    @pytest.mark.parametrize("train_layers, exp_froz_params", [(0, 53), (1, 43), (2, 24), (3, 11), (4, 1), (5, 0)])
    def test_resnet_fpn_backbone_frozen_layers(self, train_layers, exp_froz_params):
        # we know how many initial layers and parameters of the network should
        # be frozen for each trainable_backbone_layers parameter value
        # i.e. all 53 params are frozen if trainable_backbone_layers=0
        # ad first 24 params are frozen if trainable_backbone_layers=2
        model = backbone_utils.resnet_fpn_backbone("resnet50", weights=None, trainable_layers=train_layers)
        # boolean list that is true if the param at that index is frozen
        is_frozen = [not parameter.requires_grad for _, parameter in model.named_parameters()]
        # check that expected initial number of layers are frozen
        assert all(is_frozen[:exp_froz_params])

    def test_validate_resnet_inputs_detection(self):
        # default number of backbone layers to train
        ret = backbone_utils._validate_trainable_layers(
            is_trained=True, trainable_backbone_layers=None, max_value=5, default_value=3
        )
        assert ret == 3
        # can't go beyond 5
        with pytest.raises(ValueError, match=r"Trainable backbone layers should be in the range"):
            ret = backbone_utils._validate_trainable_layers(
                is_trained=True, trainable_backbone_layers=6, max_value=5, default_value=3
            )
        # if not trained, should use all trainable layers and warn
        with pytest.warns(UserWarning):
            ret = backbone_utils._validate_trainable_layers(
                is_trained=False, trainable_backbone_layers=0, max_value=5, default_value=3
            )
        assert ret == 5

    def test_transform_copy_targets(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
        targets = [{"boxes": torch.rand(3, 4)}, {"boxes": torch.rand(2, 4)}]
        targets_copy = copy.deepcopy(targets)
        out = transform(image, targets)  # noqa: F841
        assert_equal(targets[0]["boxes"], targets_copy[0]["boxes"])
        assert_equal(targets[1]["boxes"], targets_copy[1]["boxes"])

    def test_not_float_normalize(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.randint(0, 255, (3, 200, 300), dtype=torch.uint8)]
        targets = [{"boxes": torch.rand(3, 4)}]
        with pytest.raises(TypeError):
            out = transform(image, targets)  # noqa: F841

    def test_rpn_anchor_count_mismatch(self):
        # Anchor generator with a different number of anchors per location across
        # feature levels: level 0 has 2 sizes * 3 aspect ratios = 6 anchors per
        # location, while level 1 has 1 size * 3 aspect ratios = 3 anchors per
        # location.
        anchor_generator = AnchorGenerator(
            sizes=((32, 64), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0)),
        )

        # The RPN head is built using only the number of anchors of the first
        # feature level, so the predictions at the second level will not match
        # the number of anchors generated there.
        in_channels = 4
        rpn_head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )
        rpn.eval()

        images = ImageList(torch.rand(1, 3, 32, 32), [(32, 32)])
        features = OrderedDict(
            [
                ("0", torch.rand(1, in_channels, 8, 8)),
                ("1", torch.rand(1, in_channels, 4, 4)),
            ]
        )

        with pytest.raises(ValueError, match=r"(?i)anchor"):
            rpn(images, features)

    def test_rpn_anchor_count_mismatch_per_level_cancellation(self):
        # Per-level anchor counts differ between the anchor generator and the
        # RPN head, but the total counts coincide because the weighted
        # differences cancel out. The RPN head predicts 3 anchors per location
        # on every level, while the anchor generator produces [3, 2, 7] anchors
        # per location. With feature-map areas [64, 16, 4]:
        #   predictions = 3*64 + 3*16 + 3*4 = 252
        #   anchors     = 3*64 + 2*16 + 7*4 = 252
        # An aggregate total-count check would pass, but predictions would be
        # paired with the wrong level's anchors.
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128), (256, 512), (1024, 2048, 4096, 8192, 16384, 32768, 65536)),
            aspect_ratios=((1.0,), (1.0,), (1.0,)),
        )

        in_channels = 4
        rpn_head = RPNHead(in_channels, 3)

        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )
        rpn.eval()

        # Stub the proposal filtering tail so that, if the per-level anchor
        # validation is missing, execution does not fall through into NMS /
        # native ops. The real anchor generation, RPN head predictions,
        # per-level count preparation and the aggregate count guard still run.
        rpn.filter_proposals = lambda proposals, objectness, image_shapes, num_anchors_per_level: (
            [torch.empty((0, 4), device=proposals.device) for _ in image_shapes],
            [torch.empty((0,), device=proposals.device) for _ in image_shapes],
        )

        images = ImageList(torch.rand(1, 3, 64, 64), [(64, 64)])
        features = OrderedDict(
            [
                ("0", torch.rand(1, in_channels, 8, 8)),
                ("1", torch.rand(1, in_channels, 4, 4)),
                ("2", torch.rand(1, in_channels, 2, 2)),
            ]
        )

        with pytest.raises(ValueError, match=r"(?i)anchor"):
            rpn(images, features)

    def test_rpn_anchor_level_count_mismatch(self):
        # The anchor generator is configured for 2 feature levels, but 3
        # feature maps are passed to the RPN. The level-count mismatch should
        # surface as a clear ValueError rather than the anchor generator's
        # internal assertion.
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,)),
            aspect_ratios=((1.0,), (1.0,)),
        )

        in_channels = 4
        rpn_head = RPNHead(in_channels, 1)

        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )
        rpn.eval()

        images = ImageList(torch.rand(1, 3, 32, 32), [(32, 32)])
        features = OrderedDict(
            [
                ("0", torch.rand(1, in_channels, 8, 8)),
                ("1", torch.rand(1, in_channels, 4, 4)),
                ("2", torch.rand(1, in_channels, 2, 2)),
            ]
        )

        with pytest.raises(ValueError, match=r"(?i)anchor"):
            rpn(images, features)


if __name__ == "__main__":
    pytest.main([__file__])
