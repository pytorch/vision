from collections import OrderedDict

import pytest
import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from common_utils import set_rng_seed
from torchvision import models, ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
import torchvision.ops.onnx_ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# In environments without onnxruntime we prefer to
# invoke all tests in the repo and have this one skipped rather than fail.
onnxruntime = pytest.importorskip("onnxruntime")


class TestPT2ONNXExporter:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)
        onnxruntime.set_seed(42)

    def run_model(
        self,
        model: torch.nn.Module,
        inputs,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
    ):
        onnx_program = torch.onnx.export(
            model,
            inputs[0],
            verbose=False,
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            custom_translation_table=torchvision.ops.onnx_ops.onnx_translation_table(),
        )
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program)
        if len(inputs) > 1:
            for input in inputs[1:]:
                onnx_testing.assert_onnx_program(onnx_program, args=input)

    def test_nms(self):
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return ops.nms(boxes, scores, 0.5)

        self.run_model(Module(), [(boxes, scores)])

    def test_batched_nms(self):
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)
        idxs = torch.randint(0, 5, size=(num_boxes,))

        class Module(torch.nn.Module):
            def forward(self, boxes, scores, idxs):
                return ops.batched_nms(boxes, scores, idxs, 0.5)

        self.run_model(Module(), [(boxes, scores, idxs)])

    def test_clip_boxes_to_image(self):
        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)

        size_2 = torch.randn(300, 400)

        class Module(torch.nn.Module):
            def forward(self, boxes, size):
                return ops.boxes.clip_boxes_to_image(boxes, size.shape)

        self.run_model(
            Module(),
            [(boxes, size), (boxes, size_2)],
            input_names=["boxes", "size"],
            dynamic_axes={"size": [0, 1]},
        )

    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 2)
        self.run_model(model, [(x, single_roi)])

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, -1)
        self.run_model(model, [(x, single_roi)])

    def test_roi_align_aligned(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 2, aligned=True)
        self.run_model(model, [(x, single_roi)])

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_model(model, [(x, single_roi)])

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_model(model, [(x, single_roi)])

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model = ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_model(model, [(x, single_roi)])

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model = ops.RoIAlign((2, 2), 2.5, -1, aligned=True)
        self.run_model(model, [(x, single_roi)])

    def test_roi_align_malformed_boxes(self):
        x = torch.randn(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 2, 0.3, 1.5, 1.5]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 1, aligned=True)
        self.run_model(model, [(x, single_roi)])

    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2)
        self.run_model(model, [(x, rois)])

    def test_resize_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super().__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()

            def forward(self_module, images):
                return self_module.transform.resize(images, None)[0]

        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_model(
            TransformModule(),
            [(input,), (input_test,)],
            input_names=["input1"],
            dynamic_axes={"input1": [0, 1, 2]},
        )

    def test_transform_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super().__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()

            def forward(self_module, images):
                return self_module.transform(images)[0].tensors

        input = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        input_test = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        self.run_model(TransformModule(), [(input,), (input_test,)])

    def _init_test_generalized_rcnn_transform(self):
        min_size = 100
        max_size = 200
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        return transform

    def _init_test_rpn(self):
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = 256
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        return rpn

    def _init_test_roi_heads_faster_rcnn(self):
        out_channels = 256
        num_classes = 91

        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100

        box_roi_pool = ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        return roi_heads

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ("0", torch.rand(2, 256, s0 // 4, s1 // 4)),
            ("1", torch.rand(2, 256, s0 // 8, s1 // 8)),
            ("2", torch.rand(2, 256, s0 // 16, s1 // 16)),
            ("3", torch.rand(2, 256, s0 // 32, s1 // 32)),
            ("4", torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    def test_multi_scale_roi_align(self):
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = ops.MultiScaleRoIAlign(["feat1", "feat2"], 3, 2)
                self.image_sizes = [(512, 512)]

            def forward(self, input, boxes):
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i["feat1"] = torch.rand(1, 5, 64, 64)
        i["feat2"] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1["feat1"] = torch.rand(1, 5, 64, 64)
        i1["feat2"] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_model(
            TransformModule(),
            [
                (
                    i,
                    [boxes],
                ),
                (
                    i1,
                    [boxes1],
                ),
            ],
        )

    def get_image(self, rel_path: str, size: tuple[int, int]) -> torch.Tensor:
        import os

        from PIL import Image
        from torchvision.transforms import functional as F

        data_dir = os.path.join(os.path.dirname(__file__), "assets")
        path = os.path.join(data_dir, *rel_path.split("/"))
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)

        return F.convert_image_dtype(F.pil_to_tensor(image))

    def get_test_images(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            [self.get_image("encode_jpeg/grace_hopper_517x606.jpg", (100, 320))],
            [self.get_image("fakedata/logos/rgb_pytorch.png", (250, 380))],
        )

    def test_faster_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(
            weights=models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=200,
            max_size=300,
        )
        model.eval()
        model(images)
        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
        )
        # Test exported model for an image with no detections on other images
        self.run_model(
            model,
            [(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
        )

    def test_mask_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(
            weights=models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=200,
            max_size=300,
        )
        model.eval()
        model(images)
        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
        )
        # Test exported model for an image with no detections on other images
        self.run_model(
            model,
            [(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
        )

    def test_keypoint_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_images = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(
            weights=models.detection.keypoint_rcnn.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=200,
            max_size=300,
        )
        model.eval()
        model(images)
        self.run_model(
            model,
            [(images,), (test_images,), (dummy_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
        )

        self.run_model(
            model,
            [(dummy_images,), (test_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
        )

    def test_shufflenet_v2_dynamic_axes(self):
        model = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        )
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.cat([dummy_input, dummy_input, dummy_input], 0)

        self.run_model(
            model,
            [(dummy_input,), (test_inputs,)],
            input_names=["input_images"],
            output_names=["output"],
            dynamic_axes={
                "input_images": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )


if __name__ == "__main__":
    pytest.main([__file__])
