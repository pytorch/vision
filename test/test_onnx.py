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

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

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
        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io,
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

    @unittest.skip("Disable test until Split w/ zero sizes is implemented in ORT")
    def test_new_empty_tensor(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.conv2 = ops.misc.ConvTranspose2d(16, 33, (3, 5))

            def forward(self, input2):
                return self.conv2(input2)

        input = torch.rand(0, 16, 10, 10)
        test_input = torch.rand(0, 16, 20, 20)
        self.run_model(Module(), [(input, ), (test_input,)], do_constant_folding=False)

    def test_nms(self):
        boxes = torch.rand(5, 4)
        boxes[:, 2:] += torch.rand(5, 2)
        scores = torch.randn(5)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return ops.nms(boxes, scores, 0.5)

        self.run_model(Module(), [(boxes, scores)])

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

    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2)
        self.run_model(model, [(x, rois)])

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

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
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
        image = self.get_image_from_url(url=image_url, size=(200, 300))

        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url(url=image_url2, size=(250, 200))

        images = [image]
        test_images = [image2]
        return images, test_images

    def test_faster_rcnn(self):
        images, test_images = self.get_test_images()

        model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
        model.eval()
        model(images)
        self.run_model(model, [(images,), (test_images,)], input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2, 3], "outputs": [0, 1, 2, 3]},
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

    @unittest.skip("Disable test until export of interpolate script module to ONNX is fixed")
    def test_mask_rcnn(self):
        images, test_images = self.get_test_images()

        model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
        model.eval()
        model(images)
        self.run_model(model, [(images,), (test_images,)],
                       input_names=["images_tensors"],
                       output_names=["outputs"],
                       dynamic_axes={"images_tensors": [0, 1, 2, 3], "outputs": [0, 1, 2, 3]},
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

    @unittest.skip("Disable test until export of interpolate script module to ONNX is fixed")
    def test_keypoint_rcnn(self):
        class KeyPointRCNN(torch.nn.Module):
            def __init__(self):
                super(KeyPointRCNN, self).__init__()
                self.model = models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(
                    pretrained=True, min_size=200, max_size=300)

            def forward(self, images):
                output = self.model(images)
                # TODO: The keypoints_scores require the use of Argmax that is updated in ONNX.
                #       For now we are testing all the output of KeypointRCNN except keypoints_scores.
                #       Enable When Argmax is updated in ONNX Runtime.
                return output[0]['boxes'], output[0]['labels'], output[0]['scores'], output[0]['keypoints']

        images, test_images = self.get_test_images()
        model = KeyPointRCNN()
        model.eval()
        model(images)
        self.run_model(model, [(images,), (test_images,)],
                       input_names=["images_tensors"],
                       output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                       dynamic_axes={"images_tensors": [0, 1, 2, 3]},
                       tolerate_small_mismatch=True)


if __name__ == '__main__':
    unittest.main()
