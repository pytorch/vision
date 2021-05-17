import sys
from common_utils import TestCase, map_nested_tensor_object, freeze_rng_state, set_rng_seed, IN_CIRCLE_CI
from collections import OrderedDict
from itertools import product
import functools
import operator
import torch
import torch.nn as nn
from torchvision import models
import unittest
import warnings

import pytest


def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_model_unwrapper = {
    'googlenet': lambda x: x.logits,
    'inception_v3': lambda x: x.logits,
    "fasterrcnn_resnet50_fpn": lambda x: x[1],
    "fasterrcnn_mobilenet_v3_large_fpn": lambda x: x[1],
    "fasterrcnn_mobilenet_v3_large_320_fpn": lambda x: x[1],
    "maskrcnn_resnet50_fpn": lambda x: x[1],
    "keypointrcnn_resnet50_fpn": lambda x: x[1],
    "retinanet_resnet50_fpn": lambda x: x[1],
    "ssd300_vgg16": lambda x: x[1],
    "ssdlite320_mobilenet_v3_large": lambda x: x[1],
}


# The following models exhibit flaky numerics under autocast in _test_*_model harnesses.
# This may be caused by the harness environment (e.g. num classes, input initialization
# via torch.rand), and does not prove autocast is unsuitable when training with real data
# (autocast has been used successfully with real data for some of these models).
# TODO:  investigate why autocast numerics are flaky in the harnesses.
#
# For the following models, _test_*_model harnesses skip numerical checks on outputs when
# trying autocast. However, they still try an autocasted forward pass, so they still ensure
# autocast coverage suffices to prevent dtype errors in each model.
autocast_flaky_numerics = (
    "inception_v3",
    "resnet101",
    "resnet152",
    "wide_resnet101_2",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large",
    "maskrcnn_resnet50_fpn",
)


class ModelTester(TestCase):
    def _test_classification_model(self, name, input_shape, dev):
        set_rng_seed(0)
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.__dict__[name](num_classes=50)
        model.eval().to(device=dev)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        x = torch.rand(input_shape).to(device=dev)
        out = model(x)
        self.assertExpected(out.cpu(), name, prec=0.1)
        self.assertEqual(out.shape[-1], 50)
        self.check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))

        if dev == torch.device("cuda"):
            with torch.cuda.amp.autocast():
                out = model(x)
                # See autocast_flaky_numerics comment at top of file.
                if name not in autocast_flaky_numerics:
                    self.assertExpected(out.cpu(), name, prec=0.1)
                self.assertEqual(out.shape[-1], 50)

    def _test_segmentation_model(self, name, dev):
        set_rng_seed(0)
        # passing num_classes equal to a number other than 21 helps in making the test's
        # expected file size smaller
        model = models.segmentation.__dict__[name](num_classes=10, pretrained_backbone=False)
        model.eval().to(device=dev)
        input_shape = (1, 3, 32, 32)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        x = torch.rand(input_shape).to(device=dev)
        out = model(x)["out"]

        def check_out(out):
            prec = 0.01
            try:
                # We first try to assert the entire output if possible. This is not
                # only the best way to assert results but also handles the cases
                # where we need to create a new expected result.
                self.assertExpected(out.cpu(), name, prec=prec)
            except AssertionError:
                # Unfortunately some segmentation models are flaky with autocast
                # so instead of validating the probability scores, check that the class
                # predictions match.
                expected_file = self._get_expected_file(name)
                expected = torch.load(expected_file)
                self.assertEqual(out.argmax(dim=1), expected.argmax(dim=1), prec=prec)
                return False  # Partial validation performed

            return True  # Full validation performed

        full_validation = check_out(out)

        self.check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))

        if dev == torch.device("cuda"):
            with torch.cuda.amp.autocast():
                out = model(x)["out"]
                # See autocast_flaky_numerics comment at top of file.
                if name not in autocast_flaky_numerics:
                    full_validation &= check_out(out)

        if not full_validation:
            msg = "The output of {} could only be partially validated. " \
                  "This is likely due to unit-test flakiness, but you may " \
                  "want to do additional manual checks if you made " \
                  "significant changes to the codebase.".format(self._testMethodName)
            warnings.warn(msg, RuntimeWarning)
            raise unittest.SkipTest(msg)

    def _test_detection_model(self, name, dev):
        set_rng_seed(0)
        kwargs = {}
        if "retinanet" in name:
            # Reduce the default threshold to ensure the returned boxes are not empty.
            kwargs["score_thresh"] = 0.01
        elif "fasterrcnn_mobilenet_v3_large" in name:
            kwargs["box_score_thresh"] = 0.02076
            if "fasterrcnn_mobilenet_v3_large_320_fpn" in name:
                kwargs["rpn_pre_nms_top_n_test"] = 1000
                kwargs["rpn_post_nms_top_n_test"] = 1000
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False, **kwargs)
        model.eval().to(device=dev)
        input_shape = (3, 300, 300)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        x = torch.rand(input_shape).to(device=dev)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)

        def check_out(out):
            self.assertEqual(len(out), 1)

            def compact(tensor):
                size = tensor.size()
                elements_per_sample = functools.reduce(operator.mul, size[1:], 1)
                if elements_per_sample > 30:
                    return compute_mean_std(tensor)
                else:
                    return subsample_tensor(tensor)

            def subsample_tensor(tensor):
                num_elems = tensor.size(0)
                num_samples = 20
                if num_elems <= num_samples:
                    return tensor

                ith_index = num_elems // num_samples
                return tensor[ith_index - 1::ith_index]

            def compute_mean_std(tensor):
                # can't compute mean of integral tensor
                tensor = tensor.to(torch.double)
                mean = torch.mean(tensor)
                std = torch.std(tensor)
                return {"mean": mean, "std": std}

            output = map_nested_tensor_object(out, tensor_map_fn=compact)
            prec = 0.01
            try:
                # We first try to assert the entire output if possible. This is not
                # only the best way to assert results but also handles the cases
                # where we need to create a new expected result.
                self.assertExpected(output, name, prec=prec)
            except AssertionError:
                # Unfortunately detection models are flaky due to the unstable sort
                # in NMS. If matching across all outputs fails, use the same approach
                # as in NMSTester.test_nms_cuda to see if this is caused by duplicate
                # scores.
                expected_file = self._get_expected_file(name)
                expected = torch.load(expected_file)
                self.assertEqual(output[0]["scores"], expected[0]["scores"], prec=prec)

                # Note: Fmassa proposed turning off NMS by adapting the threshold
                # and then using the Hungarian algorithm as in DETR to find the
                # best match between output and expected boxes and eliminate some
                # of the flakiness. Worth exploring.
                return False  # Partial validation performed

            return True  # Full validation performed

        full_validation = check_out(out)
        self.check_jit_scriptable(model, ([x],), unwrapper=script_model_unwrapper.get(name, None))

        if dev == torch.device("cuda"):
            with torch.cuda.amp.autocast():
                out = model(model_input)
                # See autocast_flaky_numerics comment at top of file.
                if name not in autocast_flaky_numerics:
                    full_validation &= check_out(out)

        if not full_validation:
            msg = "The output of {} could only be partially validated. " \
                  "This is likely due to unit-test flakiness, but you may " \
                  "want to do additional manual checks if you made " \
                  "significant changes to the codebase.".format(self._testMethodName)
            warnings.warn(msg, RuntimeWarning)
            raise unittest.SkipTest(msg)

    def _test_detection_model_validation(self, name):
        set_rng_seed(0)
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        input_shape = (3, 300, 300)
        x = [torch.rand(input_shape)]

        # validate that targets are present in training
        self.assertRaises(ValueError, model, x)

        # validate type
        targets = [{'boxes': 0.}]
        self.assertRaises(ValueError, model, x, targets=targets)

        # validate boxes shape
        for boxes in (torch.rand((4,)), torch.rand((1, 5))):
            targets = [{'boxes': boxes}]
            self.assertRaises(ValueError, model, x, targets=targets)

        # validate that no degenerate boxes are present
        boxes = torch.tensor([[1, 3, 1, 4], [2, 4, 3, 4]])
        targets = [{'boxes': boxes}]
        self.assertRaises(ValueError, model, x, targets=targets)

    def _test_video_model(self, name, dev):
        # the default input shape is
        # bs * num_channels * clip_len * h *w
        input_shape = (1, 3, 4, 112, 112)
        # test both basicblock and Bottleneck
        model = models.video.__dict__[name](num_classes=50)
        model.eval().to(device=dev)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        x = torch.rand(input_shape).to(device=dev)
        out = model(x)
        self.check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))
        self.assertEqual(out.shape[-1], 50)

        if dev == torch.device("cuda"):
            with torch.cuda.amp.autocast():
                out = model(x)
                self.assertEqual(out.shape[-1], 50)

    def _make_sliced_model(self, model, stop_layer):
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            if name == stop_layer:
                break
        new_model = torch.nn.Sequential(layers)
        return new_model

    def test_memory_efficient_densenet(self):
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)

        for name in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
            model1 = models.__dict__[name](num_classes=50, memory_efficient=True)
            params = model1.state_dict()
            num_params = sum([x.numel() for x in model1.parameters()])
            model1.eval()
            out1 = model1(x)
            out1.sum().backward()
            num_grad = sum([x.grad.numel() for x in model1.parameters() if x.grad is not None])

            model2 = models.__dict__[name](num_classes=50, memory_efficient=False)
            model2.load_state_dict(params)
            model2.eval()
            out2 = model2(x)

            max_diff = (out1 - out2).abs().max()

            self.assertTrue(num_params == num_grad)
            self.assertTrue(max_diff < 1e-5)

    def test_resnet_dilation(self):
        # TODO improve tests to also check that each layer has the right dimensionality
        for i in product([False, True], [False, True], [False, True]):
            model = models.__dict__["resnet50"](replace_stride_with_dilation=i)
            model = self._make_sliced_model(model, stop_layer="layer4")
            model.eval()
            x = torch.rand(1, 3, 224, 224)
            out = model(x)
            f = 2 ** sum(i)
            self.assertEqual(out.shape, (1, 2048, 7 * f, 7 * f))

    def test_mobilenet_v2_residual_setting(self):
        model = models.__dict__["mobilenet_v2"](inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape[-1], 1000)

    def test_mobilenet_norm_layer(self):
        for name in ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]:
            model = models.__dict__[name]()
            self.assertTrue(any(isinstance(x, nn.BatchNorm2d) for x in model.modules()))

            def get_gn(num_channels):
                return nn.GroupNorm(32, num_channels)

            model = models.__dict__[name](norm_layer=get_gn)
            self.assertFalse(any(isinstance(x, nn.BatchNorm2d) for x in model.modules()))
            self.assertTrue(any(isinstance(x, nn.GroupNorm) for x in model.modules()))

    def test_inception_v3_eval(self):
        # replacement for models.inception_v3(pretrained=True) that does not download weights
        kwargs = {}
        kwargs['transform_input'] = True
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        name = "inception_v3"
        model = models.Inception3(**kwargs)
        model.aux_logits = False
        model.AuxLogits = None
        model = model.eval()
        x = torch.rand(1, 3, 299, 299)
        self.check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))

    def test_fasterrcnn_double(self):
        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
        model.double()
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape, dtype=torch.float64)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])

    def test_googlenet_eval(self):
        # replacement for models.googlenet(pretrained=True) that does not download weights
        kwargs = {}
        kwargs['transform_input'] = True
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        name = "googlenet"
        model = models.GoogLeNet(**kwargs)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        model = model.eval()
        x = torch.rand(1, 3, 224, 224)
        self.check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))

    @unittest.skipIf(not torch.cuda.is_available(), 'needs GPU')
    def test_fasterrcnn_switch_devices(self):
        def checkOut(out):
            self.assertEqual(len(out), 1)
            self.assertTrue("boxes" in out[0])
            self.assertTrue("scores" in out[0])
            self.assertTrue("labels" in out[0])

        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
        model.cuda()
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape, device='cuda')
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)

        checkOut(out)

        with torch.cuda.amp.autocast():
            out = model(model_input)

        checkOut(out)

        # now switch to cpu and make sure it works
        model.cpu()
        x = x.cpu()
        out_cpu = model([x])

        checkOut(out_cpu)

    def test_generalizedrcnn_transform_repr(self):

        min_size, max_size = 224, 299
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        t = models.detection.transform.GeneralizedRCNNTransform(min_size=min_size,
                                                                max_size=max_size,
                                                                image_mean=image_mean,
                                                                image_std=image_std)

        # Check integrity of object __repr__ attribute
        expected_string = 'GeneralizedRCNNTransform('
        _indent = '\n    '
        expected_string += '{0}Normalize(mean={1}, std={2})'.format(_indent, image_mean, image_std)
        expected_string += '{0}Resize(min_size=({1},), max_size={2}, '.format(_indent, min_size, max_size)
        expected_string += "mode='bilinear')\n)"
        self.assertEqual(t.__repr__(), expected_string)


_devs = [torch.device("cpu"), torch.device("cuda")] if torch.cuda.is_available() else [torch.device("cpu")]


@pytest.mark.parametrize('model_name', get_available_classification_models())
@pytest.mark.parametrize('dev', _devs)
def test_classification_model(model_name, dev):
    input_shape = (1, 3, 299, 299) if model_name == 'inception_v3' else (1, 3, 224, 224)
    ModelTester()._test_classification_model(model_name, input_shape, dev)


@pytest.mark.parametrize('model_name', get_available_segmentation_models())
@pytest.mark.parametrize('dev', _devs)
def test_segmentation_model(model_name, dev):
    ModelTester()._test_segmentation_model(model_name, dev)


@pytest.mark.parametrize('model_name', get_available_detection_models())
@pytest.mark.parametrize('dev', _devs)
def test_detection_model(model_name, dev):
    ModelTester()._test_detection_model(model_name, dev)


@pytest.mark.parametrize('model_name', get_available_detection_models())
def test_detection_model_validation(model_name):
    ModelTester()._test_detection_model_validation(model_name)


@pytest.mark.parametrize('model_name', get_available_video_models())
@pytest.mark.parametrize('dev', _devs)
def test_video_model(model_name, dev):
    if IN_CIRCLE_CI and 'cuda' in dev.type and model_name == 'r2plus1d_18' and sys.platform == 'linux':
        # FIXME: Failure should fixed and test re-actived. See https://github.com/pytorch/vision/issues/3702
        pytest.skip('r2plus1d_18 fails on CircleCI linux GPU machines.')
    ModelTester()._test_video_model(model_name, dev)


if __name__ == '__main__':
    pytest.main([__file__])
