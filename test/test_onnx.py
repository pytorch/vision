# onnxruntime requires python 3.5 or above
try:
    # This import should be before that of torch
    # see https://github.com/onnx/onnx/issues/2394#issuecomment-581638840
    import onnxruntime
except ImportError:
    onnxruntime = None

from common_utils import set_rng_seed
import io
import torch
from torchvision import ops
from torchvision import models
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor

from collections import OrderedDict

import unittest
from torchvision.ops._register_onnx_ops import _onnx_opset_version


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False, do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        if isinstance(inputs_list[0][-1], dict):
            torch_onnx_input = inputs_list[0] + ({},)
        else:
            torch_onnx_input = inputs_list[0]
        # export to onnx with the first input
        torch.onnx.export(model, torch_onnx_input, onnx_io,
                          do_constant_folding=do_constant_folding, opset_version=_onnx_opset_version,
                          dynamic_axes=dynamic_axes, input_names=input_names, output_names=output_names)
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or \
                   isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)

        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

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

        self.run_model(Module(), [(boxes, size), (boxes, size_2)],
                       input_names=["boxes", "size"],
                       dynamic_axes={"size": [0, 1]})

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

    @unittest.skip  # Issue in exporting ROIAlign with aligned = True for malformed boxes
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
                super(TransformModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()

            def forward(self_module, images):
                return self_module.transform.resize(images, None)[0]

        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_model(TransformModule(), [(input,), (input_test,)],
                       input_names=["input1"], dynamic_axes={"input1": [0, 1, 2]})

    def test_transform_images(self):

        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super(TransformModule, self_module).__init__()
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
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)
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
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        return roi_heads

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ('0', torch.rand(2, 256, s0 // 4, s1 // 4)),
            ('1', torch.rand(2, 256, s0 // 8, s1 // 8)),
            ('2', torch.rand(2, 256, s0 // 16, s1 // 16)),
            ('3', torch.rand(2, 256, s0 // 32, s1 // 32)),
            ('4', torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    def test_rpn(self):
        set_rng_seed(0)

        class RPNModule(torch.nn.Module):
            def __init__(self_module):
                super(RPNModule, self_module).__init__()
                self_module.rpn = self._init_test_rpn()

            def forward(self_module, images, features):
                images = ImageList(images, [i.shape[-2:] for i in images])
                return self_module.rpn(images, features)

        images = torch.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)

        model = RPNModule()
        model.eval()
        model(images, features)

        self.run_model(model, [(images, features), (images2, test_features)], tolerate_small_mismatch=True,
                       input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                       dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3],
                                     "input3": [0, 1, 2, 3], "input4": [0, 1, 2, 3],
                                     "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]})

    def test_multi_scale_roi_align(self):

        class TransformModule(torch.nn.Module):
            def __init__(self):
                super(TransformModule, self).__init__()
                self.model = ops.MultiScaleRoIAlign(['feat1', 'feat2'], 3, 2)
                self.image_sizes = [(512, 512)]

            def forward(self, input, boxes):
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i['feat1'] = torch.rand(1, 5, 64, 64)
        i['feat2'] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1['feat1'] = torch.rand(1, 5, 64, 64)
        i1['feat2'] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_model(TransformModule(), [(i, [boxes],), (i1, [boxes1],)])

    def test_roi_heads(self):
        class RoiHeadsModule(torch.nn.Module):
            def __init__(self_module):
                super(RoiHeadsModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()
                self_module.rpn = self._init_test_rpn()
                self_module.roi_heads = self._init_test_roi_heads_faster_rcnn()

            def forward(self_module, images, features):
                original_image_sizes = [img.shape[-2:] for img in images]
                images = ImageList(images, [i.shape[-2:] for i in images])
                proposals, _ = self_module.rpn(images, features)
                detections, _ = self_module.roi_heads(features, proposals, images.image_sizes)
                detections = self_module.transform.postprocess(detections,
                                                               images.image_sizes,
                                                               original_image_sizes)
                return detections

        images = torch.rand(2, 3, 100, 100)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 150, 150)
        test_features = self.get_features(images2)

        model = RoiHeadsModule()
        model.eval()
        model(images, features)

        self.run_model(model, [(images, features), (images2, test_features)], tolerate_small_mismatch=True,
                       input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                       dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3], "input3": [0, 1, 2, 3],
                                     "input4": [0, 1, 2, 3], "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]})

    def get_image_from_url(self, url, size=None):
        import requests
        from PIL import Image
        from io import BytesIO
        from torchvision import transforms

        data = requests.get(url)
        image = Image.open(BytesIO(data.content)).convert("RGB")

        if size is None:
            size = (300, 200)
        image = image.resize(size, Image.BILINEAR)

        to_tensor = transforms.ToTensor()
        return to_tensor(image)

    def get_test_images(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image = self.get_image_from_url(url=image_url, size=(100, 320))

        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url(url=image_url2, size=(250, 380))

        images = [image]
        test_images = [image2]
        return images, test_images

    def test_faster_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
        model.eval()
        model(images)
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [(images,), (test_images,), (dummy_image,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)
        # Test exported model for an image with no detections on other images
        self.run_model(model, [(dummy_image,), (images,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
                       tolerate_small_mismatch=True)

    # Verify that paste_mask_in_image beahves the same in tracing.
    # This test also compares both paste_masks_in_image and _onnx_paste_masks_in_image
    # (since jit_trace witll call _onnx_paste_masks_in_image).
    def test_paste_mask_in_image(self):
        # disable profiling
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        masks = torch.rand(10, 1, 26, 26)
        boxes = torch.rand(10, 4)
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = torch.jit.trace(paste_masks_in_image,
                                    (masks, boxes,
                                     [torch.tensor(o_im_s[0]),
                                      torch.tensor(o_im_s[1])]))
        out_trace = jit_trace(masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])])

        assert torch.all(out.eq(out_trace))

        masks2 = torch.rand(20, 1, 26, 26)
        boxes2 = torch.rand(20, 4)
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])])

        assert torch.all(out2.eq(out_trace2))

    def test_mask_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
        model.eval()
        model(images)
        # Test exported model on images of different size, or dummy input
        self.run_model(model, [(images,), (test_images,), (dummy_image,)],
                       input_names=["images_tensors"],
                       output_names=["boxes", "labels", "scores", "masks"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                     "scores": [0], "masks": [0, 1, 2]},
                       tolerate_small_mismatch=True)
        # TODO: enable this test once dynamic model export is fixed
        # Test exported model for an image with no detections on other images
        self.run_model(model, [(dummy_image,), (images,)],
                       input_names=["images_tensors"],
                       output_names=["boxes", "labels", "scores", "masks"],
                       dynamic_axes={"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                     "scores": [0], "masks": [0, 1, 2]},
                       tolerate_small_mismatch=True)

    # Verify that heatmaps_to_keypoints behaves the same in tracing.
    # This test also compares both heatmaps_to_keypoints and _onnx_heatmaps_to_keypoints
    # (since jit_trace witll call _heatmaps_to_keypoints).
    # @unittest.skip("Disable test until Resize bug fixed in ORT")
    def test_heatmaps_to_keypoints(self):
        # disable profiling
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        maps = torch.rand(10, 1, 26, 26)
        rois = torch.rand(10, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)

        assert torch.all(out[0].eq(out_trace[0]))
        assert torch.all(out[1].eq(out_trace[1]))

        maps2 = torch.rand(20, 2, 21, 21)
        rois2 = torch.rand(20, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)

        assert torch.all(out2[0].eq(out_trace2[0]))
        assert torch.all(out2[1].eq(out_trace2[1]))

    def test_keypoint_rcnn(self):
        images, test_images = self.get_test_images()
        dummy_images = [torch.ones(3, 100, 100) * 0.3]
        model = models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
        model.eval()
        model(images)
        self.run_model(model, [(images,), (test_images,), (dummy_images,)],
                       input_names=["images_tensors"],
                       output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                       dynamic_axes={"images_tensors": [0, 1, 2]},
                       tolerate_small_mismatch=True)

        self.run_model(model, [(dummy_images,), (test_images,)],
                       input_names=["images_tensors"],
                       output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                       dynamic_axes={"images_tensors": [0, 1, 2]},
                       tolerate_small_mismatch=True)

    def test_shufflenet_v2_dynamic_axes(self):
        model = models.shufflenet_v2_x0_5(pretrained=True)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.cat([dummy_input, dummy_input, dummy_input], 0)

        self.run_model(model, [(dummy_input,), (test_inputs,)],
                       input_names=["input_images"],
                       output_names=["output"],
                       dynamic_axes={"input_images": {0: 'batch_size'}, "output": {0: 'batch_size'}},
                       tolerate_small_mismatch=True)


if __name__ == '__main__':
    unittest.main()
