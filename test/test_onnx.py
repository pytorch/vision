import io
import torch
from torchvision import ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform

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

    def run_model(self, model, inputs, other_inputs=None):
        model.eval()

        # run pytorch model
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            outputs = model(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)

        onnx_io = io.BytesIO()
        # export to onnx
        torch.onnx.export(model, inputs, onnx_io, do_constant_folding=True, opset_version=10)

        # validate the exported model with onnx runtime
        self.ort_validate(onnx_io, inputs, outputs)

        # verify with the other inputs
        if other_inputs:
            for test_inputs in other_inputs:
                with torch.no_grad():
                    if isinstance(test_inputs, torch.Tensor):
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

        self.run_model(Module(), (boxes, scores))

    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 2)
        self.run_model(model, (x, single_roi))

    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2)
        model.eval()
        self.run_model(model, (x, rois))

    @unittest.skip("Disable test until Resize opset 11 is implemented in ONNX Runtime")
    def test_transform_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super(TransformModule, self_module).__init__()
                min_size = 800
                max_size = 1333
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]
                self_module.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
            def forward(self_module, images):
                return self_module.transform(images)[0].tensors

        input = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        input_test = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        self.run_model(TransformModule(), (input,), [(input_test,)])

if __name__ == '__main__':
    unittest.main()
