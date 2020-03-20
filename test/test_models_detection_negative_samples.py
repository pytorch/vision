import torch

import torchvision.models
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

import unittest


class Tester(unittest.TestCase):

    def test_targets_to_anchors(self):
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        negative_target = {"boxes": boxes,
                           "labels": torch.zeros((1, 1), dtype=torch.int64),
                           "image_id": 4,
                           "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                           "iscrowd": torch.zeros((0,), dtype=torch.int64)}

        anchors = [torch.randint(-50, 50, (3, 4), dtype=torch.float32)]
        targets = [negative_target]

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        rpn_head = RPNHead(4, rpn_anchor_generator.num_anchors_per_location()[0])

        head = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            0.5, 0.3,
            256, 0.5,
            2000, 2000, 0.7)

        labels, matched_gt_boxes = head.assign_targets_to_anchors(anchors, targets)

        self.assertEqual(labels[0].sum(), 0)
        self.assertEqual(labels[0].shape, torch.Size([anchors[0].shape[0]]))
        self.assertEqual(labels[0].dtype, torch.float32)

        self.assertEqual(matched_gt_boxes[0].sum(), 0)
        self.assertEqual(matched_gt_boxes[0].shape, anchors[0].shape)
        self.assertEqual(matched_gt_boxes[0].dtype, torch.float32)

    def test_assign_targets_to_proposals(self):

        proposals = [torch.randint(-50, 50, (20, 4), dtype=torch.float32)]
        gt_boxes = [torch.zeros((0, 4), dtype=torch.float32)]
        gt_labels = [torch.tensor([[0]], dtype=torch.int64)]

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            4 * resolution ** 2,
            representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            2)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            0.5, 0.5,
            512, 0.25,
            None,
            0.05, 0.5, 100)

        matched_idxs, labels = roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        self.assertEqual(matched_idxs[0].sum(), 0)
        self.assertEqual(matched_idxs[0].shape, torch.Size([proposals[0].shape[0]]))
        self.assertEqual(matched_idxs[0].dtype, torch.int64)

        self.assertEqual(labels[0].sum(), 0)
        self.assertEqual(labels[0].shape, torch.Size([proposals[0].shape[0]]))
        self.assertEqual(labels[0].dtype, torch.int64)

    def test_forward_negative_sample(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        images = [torch.rand((3, 100, 100), dtype=torch.float32)]
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        negative_target = {"boxes": boxes,
                           "labels": torch.zeros((1, 1), dtype=torch.int64),
                           "image_id": 4,
                           "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                           "iscrowd": torch.zeros((0,), dtype=torch.int64)}

        targets = [negative_target]

        loss_dict = model(images, targets)

        self.assertEqual(loss_dict["loss_box_reg"], torch.tensor(0.))
        self.assertEqual(loss_dict["loss_rpn_box_reg"], torch.tensor(0.))


if __name__ == '__main__':
    unittest.main()
