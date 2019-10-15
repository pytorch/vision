from common_utils import TestCase, map_nested_tensor_object
from collections import OrderedDict
from itertools import product
import torch
import numpy as np
from torchvision import models
import unittest
import traceback
import random
import inspect


EPSILON = 1e-5  # small value for approximate comparisons/assertions
STANDARD_NUM_CLASSES = 50
STANDARD_INPUT_SHAPE = (1, 3, 224, 224)
STANDARD_SEED = 1729  # https://fburl.com/3i5wkg9p


def set_rng_seed(seed=STANDARD_SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def subsample_tensor(tensor, num_samples=20):
    num_elems = tensor.numel()
    if num_elems <= num_samples:
        return tensor

    flat_tensor = tensor.flatten()
    ith_index = num_elems // num_samples
    return flat_tensor[ith_index - 1::ith_index]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# models that are in torch hub, as well as r3d_18. we tried testing all models
# but the test was too slow. not included are detection models, because
# they are not yet supported in JIT.
script_test_models = [
    "deeplabv3_resnet101",
    "fcn_resnet101",
    'r3d_18',
]


class ModelTester(TestCase):

    # create random tensor with given shape using synced RNG state
    # caching because these tests take pretty long already (instantiating models and all)
    TEST_INPUTS = {}

    def _get_test_input(self, shape=STANDARD_INPUT_SHAPE):
        # NOTE not thread-safe, but should give same results even if multi-threaded testing gave a race condition
        # giving consistent results is kind of the point of this helper method
        if shape not in self.TEST_INPUTS:
            set_rng_seed(STANDARD_SEED)
            self.TEST_INPUTS[shape] = torch.rand(shape)
        return self.TEST_INPUTS[shape]

    # create a randomly-weighted model w/ synced RNG state
    def _get_test_model(self, callable, **kwargs):
        set_rng_seed(STANDARD_SEED)
        model = callable(**kwargs)
        model.eval()
        return model

    def check_script(self, model, name):
        if name not in script_test_models:
            return
        scriptable = True
        msg = ""
        try:
            torch.jit.script(model)
        except Exception as e:
            tb = traceback.format_exc()
            scriptable = False
            msg = str(e) + str(tb)
        self.assertTrue(scriptable, msg)

    def _check_scriptable(self, model, expected):
        if expected is None:  # we don't check scriptability for all models
            return

        actual = True
        msg = ''
        try:
            torch.jit.script(model)
        except Exception as e:
            tb = traceback.format_exc()
            actual = False
            msg = str(e) + str(tb)
        self.assertEqual(actual, expected, msg)


class ClassificationCoverageTester(TestCase):

    # Find all models exposed by torchvision.models factory methods (with assumptions)
    def get_available_classification_models(self):
        # TODO add a registration mechanism to torchvision.models
        return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

    # Recursively gather test methods from all classification testers
    def get_test_methods_for_class(self, klass):
        all_methods = inspect.getmembers(klass, predicate=inspect.isfunction)
        test_methods = set([method[0] for method in all_methods if method[0].startswith('test_')])
        for child in klass.__subclasses__():
            test_methods = test_methods.union(self.get_test_methods_for_class(child))
        return test_methods

    # Verify that all models exposed by torchvision.models factory methods
    #    have corresponding test methods
    # NOTE This does not include some of the extra tests (such as Resnet
    #    dilation) and says nothing about the correctness of the test, nor
    #    of the model. It just enforces a naming scheme on the tests, and
    #    verifies that all models have a corresponding test name.
    def test_classification_model_coverage(self):
        model_names = self.get_available_classification_models()
        test_names = self.get_test_methods_for_class(ClassificationModelTester)

        for model_name in model_names:
            test_name = 'test_' + model_name
            self.assertTrue(test_name in test_names)


class ClassificationModelTester(ModelTester):
    def _infer_for_test_with(self, model, test_input):
        return model(test_input)

    def _check_classification_output_shape(self, test_output, num_classes):
        self.assertEqual(test_output.shape, (1, num_classes))

    # NOTE Depends on presence of test data fixture. See common_utils.py for
    #    details on creating fixtures.
    def _check_model_correctness(self, model, test_input, num_classes=STANDARD_NUM_CLASSES):
        test_output = self._infer_for_test_with(model, test_input)
        self._check_classification_output_shape(test_output, num_classes)
        self.assertExpected(test_output)
        return test_output

    # NOTE override this in a child class
    def _get_input_shape(self):
        return STANDARD_INPUT_SHAPE

    def _test_classification_model(self, model_callable, num_classes=STANDARD_NUM_CLASSES, **kwargs):
        model = self._get_test_model(model_callable, num_classes=num_classes, **kwargs)
        self._check_scriptable(model, True)  # currently, all expected to be scriptable
        test_input = self._get_test_input(shape=self._get_input_shape())
        test_output = self._check_model_correctness(model, test_input)
        return model, test_input, test_output


class AlexnetTester(ClassificationModelTester):
    def test_alexnet(self):
        self._test_classification_model(models.alexnet)


# TODO add test for aux_logits arg to factory method
# TODO add test for transform_input arg to factory method
class InceptionV3Tester(ClassificationModelTester):
    def _get_input_shape(self):
        return (1, 3, 299, 299)

    def test_inception_v3(self):
        self._test_classification_model(models.inception_v3)


class SqueezenetTester(ClassificationModelTester):
    def test_squeezenet1_0(self):
        self._test_classification_model(models.squeezenet1_0)

    def test_squeezenet1_1(self):
        self._test_classification_model(models.squeezenet1_1)


# TODO add test for width_mult arg to factory method
class MobilenetTester(ClassificationModelTester):
    def test_mobilenet_v2(self):
        self._test_classification_model(models.mobilenet_v2)

    def test_mobilenetv2_residual_setting(self):
        self._test_classification_model(models.mobilenet_v2, inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])


# TODO add test for aux_logits arg to factory method
# TODO add test for transform_input arg to factory method
class GooglenetTester(ClassificationModelTester):
    def test_googlenet(self):
        self._test_classification_model(models.googlenet)


class VGGNetTester(ClassificationModelTester):
    def test_vgg11(self):
        self._test_classification_model(models.vgg11)

    def test_vgg11_bn(self):
        self._test_classification_model(models.vgg11_bn)

    def test_vgg13(self):
        self._test_classification_model(models.vgg13)

    def test_vgg13_bn(self):
        self._test_classification_model(models.vgg13_bn)

    def test_vgg16(self):
        self._test_classification_model(models.vgg16)

    def test_vgg16_bn(self):
        self._test_classification_model(models.vgg16_bn)

    def test_vgg19(self):
        self._test_classification_model(models.vgg19)

    def test_vgg19_bn(self):
        self._test_classification_model(models.vgg19_bn)


# TODO add test for dropout arg to factory method
class MNASNetTester(ClassificationModelTester):
    def test_mnasnet0_5(self):
        self._test_classification_model(models.mnasnet0_5)

    def test_mnasnet0_75(self):
        self._test_classification_model(models.mnasnet0_75)

    def test_mnasnet1_0(self):
        self._test_classification_model(models.mnasnet1_0)

    def test_mnasnet1_3(self):
        self._test_classification_model(models.mnasnet1_3)


# TODO add test for bn_size arg to factory method
# TODO add test for drop_rate arg to factory method
class DensenetTester(ClassificationModelTester):
    def _test_densenet_plus_mem_eff(self, model_callable):
        model, test_input, test_output = self._test_classification_model(model_callable)

        # above, we perform the standard correctness test against the test fixture, and capture key test params
        # below, we check that memory efficient/time inefficient DenseNet implementation behaves like the "standard" one
        me_model = self._get_test_model(model_callable, num_classes=STANDARD_NUM_CLASSES, memory_efficient=True)
        me_model.load_state_dict(model.state_dict())  # xfer weights over
        me_output = self._infer_for_test_with(me_model, test_input)
        test_output.squeeze(0)
        me_output.squeeze(0)
        self.assertTrue((test_output - me_output).abs().max() < EPSILON)

    def test_densenet121(self):
        self._test_densenet_plus_mem_eff(models.densenet121)

    def test_densenet161(self):
        self._test_densenet_plus_mem_eff(models.densenet161)

    def test_densenet169(self):
        self._test_densenet_plus_mem_eff(models.densenet169)

    def test_densenet201(self):
        self._test_densenet_plus_mem_eff(models.densenet201)


class ShufflenetTester(ClassificationModelTester):
    def test_shufflenet_v2_x0_5(self):
        self._test_classification_model(models.shufflenet_v2_x0_5)

    def test_shufflenet_v2_x1_0(self):
        self._test_classification_model(models.shufflenet_v2_x1_0)

    def test_shufflenet_v2_x1_5(self):
        self._test_classification_model(models.shufflenet_v2_x1_5)

    def test_shufflenet_v2_x2_0(self):
        self._test_classification_model(models.shufflenet_v2_x2_0)


# TODO add test for zero_init_residual arg to factory method
# TODO add test for norm_layer arg to factory method
class ResnetTester(ClassificationModelTester):
    def _get_scriptability_value(self):
        return True

    def test_resnet18(self):
        self._test_classification_model(models.resnet18)

    def test_resnet34(self):
        self._test_classification_model(models.resnet34)

    def test_resnet50(self):
        self._test_classification_model(models.resnet50)

    def test_resnet101(self):
        self._test_classification_model(models.resnet101)

    def test_resnet152(self):
        self._test_classification_model(models.resnet152)

    def test_resnext50_32x4d(self):
        self._test_classification_model(models.resnext50_32x4d)

    def test_resnext101_32x8d(self):
        self._test_classification_model(models.resnext101_32x8d)

    def test_wide_resnet50_2(self):
        self._test_classification_model(models.wide_resnet50_2)

    def test_wide_resnet101_2(self):
        self._test_classification_model(models.wide_resnet101_2)

    def _make_sliced_model(self, model, stop_layer):
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            if name == stop_layer:
                break
        new_model = torch.nn.Sequential(layers)
        return new_model

    def test_resnet_dilation(self):
        # TODO improve tests to also check that each layer has the right dimensionality
        for i in product([False, True], [False, True], [False, True]):
            model = models.__dict__["resnet50"](replace_stride_with_dilation=i)
            model = self._make_sliced_model(model, stop_layer="layer4")
            model.eval()
            x = self._get_test_input(STANDARD_INPUT_SHAPE)
            out = model(x)
            f = 2 ** sum(i)
            self.assertEqual(out.shape, (1, 2048, 7 * f, 7 * f))


class SegmentationModelTester(ModelTester):
    def _test_segmentation_model(self, name):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.segmentation.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(tuple(out["out"].shape), (1, 50, 300, 300))


class DetectionModelTester(ModelTester):
    def _test_detection_model(self, name):
        set_rng_seed(0)
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)

        def compute_mean_std(tensor):
            # can't compute mean of integral tensor
            tensor = tensor.to(torch.double)
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            return {"mean": mean, "std": std}

        # maskrcnn_resnet_50_fpn numerically unstable across platforms, so for now
        # compare results with mean and std
        if name == "maskrcnn_resnet50_fpn":
            test_value = map_nested_tensor_object(out, tensor_map_fn=compute_mean_std)
            # mean values are small, use large rtol
            self.assertExpected(test_value, rtol=.01, atol=.01)
        else:
            self.assertExpected(map_nested_tensor_object(out, tensor_map_fn=subsample_tensor))

        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])

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


class VideoModelTester(ModelTester):
    def _test_video_model(self, name):
        # the default input shape is
        # bs * num_channels * clip_len * h *w
        input_shape = (1, 3, 4, 112, 112)
        # test both basicblock and Bottleneck
        model = models.video.__dict__[name](num_classes=50)
        self.check_script(model, name)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)


for model_name in get_available_segmentation_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_segmentation_model(model_name)

    setattr(SegmentationModelTester, "test_" + model_name, do_test)


for model_name in get_available_detection_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_detection_model(model_name)

    setattr(DetectionModelTester, "test_" + model_name, do_test)


for model_name in get_available_video_models():

    def do_test(self, model_name=model_name):
        self._test_video_model(model_name)

    setattr(VideoModelTester, "test_" + model_name, do_test)

if __name__ == '__main__':
    unittest.main()
