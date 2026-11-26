import pytest
import torch
import torchvision.models
from common_utils import assert_equal
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign


class TestModelsDetectionNegativeSamples:
    def _make_empty_sample(self, add_masks=False, add_keypoints=False):
        images = [torch.rand((3, 100, 100), dtype=torch.float32)]
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        negative_target = {
            "boxes": boxes,
            "labels": torch.zeros(0, dtype=torch.int64),
            "image_id": 4,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }

        if add_masks:
            negative_target["masks"] = torch.zeros(0, 100, 100, dtype=torch.uint8)

        if add_keypoints:
            negative_target["keypoints"] = torch.zeros(17, 0, 3, dtype=torch.float32)

        targets = [negative_target]
        return images, targets

    def test_targets_to_anchors(self):
        _, targets = self._make_empty_sample()
        anchors = [torch.randint(-50, 50, (3, 4), dtype=torch.float32)]

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(4, rpn_anchor_generator.num_anchors_per_location()[0])

        head = RegionProposalNetwork(rpn_anchor_generator, rpn_head, 0.5, 0.3, 256, 0.5, 2000, 2000, 0.7, 0.05)

        labels, matched_gt_boxes = head.assign_targets_to_anchors(anchors, targets)

        assert labels[0].sum() == 0
        assert labels[0].shape == torch.Size([anchors[0].shape[0]])
        assert labels[0].dtype == torch.float32

        assert matched_gt_boxes[0].sum() == 0
        assert matched_gt_boxes[0].shape == anchors[0].shape
        assert matched_gt_boxes[0].dtype == torch.float32

    def test_assign_targets_to_proposals(self):

        proposals = [torch.randint(-50, 50, (20, 4), dtype=torch.float32)]
        gt_boxes = [torch.zeros((0, 4), dtype=torch.float32)]
        gt_labels = [torch.tensor([[0]], dtype=torch.int64)]

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(4 * resolution**2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, 2)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            0.5,
            0.5,
            512,
            0.25,
            None,
            0.05,
            0.5,
            100,
        )

        matched_idxs, labels = roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        assert matched_idxs[0].sum() == 0
        assert matched_idxs[0].shape == torch.Size([proposals[0].shape[0]])
        assert matched_idxs[0].dtype == torch.int64

        assert labels[0].sum() == 0
        assert labels[0].shape == torch.Size([proposals[0].shape[0]])
        assert labels[0].dtype == torch.int64

    @pytest.mark.parametrize(
        "name",
        [
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
        ],
    )
    def test_forward_negative_sample_frcnn(self, name):
        model = torchvision.models.get_model(
            name, weights=None, weights_backbone=None, num_classes=2, min_size=100, max_size=100
        )

        images, targets = self._make_empty_sample()
        loss_dict = model(images, targets)

        assert_equal(loss_dict["loss_box_reg"], torch.tensor(0.0))
        assert_equal(loss_dict["loss_rpn_box_reg"], torch.tensor(0.0))

    def test_forward_negative_sample_mrcnn(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=100, max_size=100
        )

        images, targets = self._make_empty_sample(add_masks=True)
        loss_dict = model(images, targets)

        assert_equal(loss_dict["loss_box_reg"], torch.tensor(0.0))
        assert_equal(loss_dict["loss_rpn_box_reg"], torch.tensor(0.0))
        assert_equal(loss_dict["loss_mask"], torch.tensor(0.0))

    def test_forward_negative_sample_krcnn(self):
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=100, max_size=100
        )

        images, targets = self._make_empty_sample(add_keypoints=True)
        loss_dict = model(images, targets)

        assert_equal(loss_dict["loss_box_reg"], torch.tensor(0.0))
        assert_equal(loss_dict["loss_rpn_box_reg"], torch.tensor(0.0))
        assert_equal(loss_dict["loss_keypoint"], torch.tensor(0.0))

    def test_forward_negative_sample_retinanet(self):
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=100, max_size=100
        )

        images, targets = self._make_empty_sample()
        loss_dict = model(images, targets)

        assert_equal(loss_dict["bbox_regression"], torch.tensor(0.0))

    def test_forward_negative_sample_fcos(self):
        model = torchvision.models.detection.fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2, min_size=100, max_size=100
        )

        images, targets = self._make_empty_sample()
        loss_dict = model(images, targets)

        assert_equal(loss_dict["bbox_regression"], torch.tensor(0.0))
        assert_equal(loss_dict["bbox_ctrness"], torch.tensor(0.0))

    def test_forward_negative_sample_ssd(self):
        model = torchvision.models.detection.ssd300_vgg16(weights=None, weights_backbone=None, num_classes=2)

        images, targets = self._make_empty_sample()
        loss_dict = model(images, targets)

        assert_equal(loss_dict["bbox_regression"], torch.tensor(0.0))

    @pytest.mark.parametrize(
        "name",
        [
            "fasterrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn",
            "keypointrcnn_resnet50_fpn",
        ],
    )
    @pytest.mark.parametrize("limit", [-1, -2])
    def test_negative_box_detections_per_img(self, name, limit):
        # A negative box_detections_per_img used to be applied silently as a
        # Python negative slice in postprocess_detections, dropping the final
        # detections instead of raising an error.
        with pytest.raises(ValueError, match="detections_per_img should be non-negative"):
            torchvision.models.get_model(
                name, weights=None, weights_backbone=None, num_classes=2, box_detections_per_img=limit
            )

    def test_roi_heads_negative_detections_per_img(self):
        # The validation lives in RoIHeads itself, so constructing one
        # directly with a negative detections_per_img must also raise,
        # without going through a model factory.
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        box_head = TwoMLPHead(4 * resolution**2, 1024)
        box_predictor = FastRCNNPredictor(1024, 2)

        with pytest.raises(ValueError, match="detections_per_img should be non-negative"):
            RoIHeads(
                box_roi_pool,
                box_head,
                box_predictor,
                0.5,
                0.5,
                512,
                0.25,
                None,
                0.05,
                0.5,
                -1,
            )

    def test_box_detections_per_img_zero_and_positive(self):
        # Based on the reproducer in https://github.com/pytorch/vision/issues/9761:
        # non-negative values must keep their previous behavior, with 0
        # returning no detections and a positive value capping the output.
        image = torch.linspace(0, 1, 3 * 64 * 64).reshape(3, 64, 64)

        def run(limit):
            torch.manual_seed(109150)
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=2,
                box_score_thresh=0.0,
                box_nms_thresh=1.0,
                box_detections_per_img=limit,
                min_size=64,
                max_size=64,
            ).eval()
            with torch.no_grad():
                return model([image])[0]

        full = run(100)
        assert len(full["scores"]) > 10

        capped = run(10)
        assert_equal(capped["boxes"], full["boxes"][:10])
        assert_equal(capped["scores"], full["scores"][:10])
        assert_equal(capped["labels"], full["labels"][:10])

        empty = run(0)
        assert len(empty["boxes"]) == 0
        assert len(empty["scores"]) == 0
        assert len(empty["labels"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
