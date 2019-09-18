import io
import torch
from torchvision import ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import unittest


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io, do_constant_folding=True, opset_version=10)

        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or \
                   isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs)

    def ort_validate(self, onnx_io, inputs, outputs):

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
            torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)

    def test_nms(self):
        boxes = torch.rand(5, 4)
        boxes[:, 2:] += torch.rand(5, 2)
        scores = torch.randn(5)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return ops.nms(boxes, scores, 0.5)

        self.run_model(Module(), [(boxes, scores)])

    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 2)
        self.run_model(model, [(x, single_roi)])

    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2)
        model.eval()
        self.run_model(model, [(x, rois)])

    @unittest.skip("Disable test until Resize opset 11 is implemented in ONNX Runtime")
    def test_transform_images(self):

        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super(TransformModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()

            def forward(self_module, images):
                return self_module.transform(images)[0].tensors

        input = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        input_test = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        self.run_model(TransformModule(), [input, input_test])

    def _init_test_generalized_rcnn_transform(self):
        min_size = 800
        max_size = 1333
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

    def get_image_from_url(self, url):
        import requests
        import numpy
        from PIL import Image
        from io import BytesIO
        from torchvision import transforms

        data = requests.get(url)
        image = Image.open(BytesIO(data.content)).convert("RGB")
        image = image.resize((800, 1280), Image.BILINEAR)

        to_tensor = transforms.ToTensor()
        return to_tensor(image)

    @unittest.skip("Disable test until Resize opset 11 is implemented in ONNX Runtime")
    def test_rpn(self):
        class RPNModule(torch.nn.Module):
            def __init__(self_module):
                super(RPNModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()
                self_module.backbone = resnet_fpn_backbone('resnet50', True)
                self_module.rpn = self._init_test_rpn()

            def forward(self_module, images):
                images, targets = self_module.transform(images)
                features = self_module.backbone(images.tensors)
                return self_module.rpn(images, features, targets)

        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image1 = self.get_image_from_url(url=image_url)
        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url(url=image_url2)
        images = [image1]
        test_images = [image2]

        model = RPNModule()
        model.eval()
        model(images)
        self.run_model(model, [(images,), (test_images,)])


if __name__ == '__main__':
    unittest.main()
