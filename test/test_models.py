import os
import io
import sys
from common_utils import map_nested_tensor_object, freeze_rng_state, set_rng_seed, cpu_and_gpu, needs_cuda
from _utils_internal import get_relative_path
from collections import OrderedDict
import functools
import operator
import torch
import torch.fx
import torch.nn as nn
import torchvision
from torchvision import models
import pytest
import warnings
import traceback


ACCEPT = os.getenv('EXPECTTEST_ACCEPT', '0') == '1'


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


def get_available_quantizable_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.quantization.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def _get_expected_file(name=None):
    # Determine expected file based on environment
    expected_file_base = get_relative_path(os.path.realpath(__file__), "expect")

    # Note: for legacy reasons, the reference file names all had "ModelTest.test_" in their names
    # We hardcode it here to avoid having to re-generate the reference files
    expected_file = expected_file = os.path.join(expected_file_base, 'ModelTester.test_' + name)
    expected_file += "_expect.pkl"

    if not ACCEPT and not os.path.exists(expected_file):
        raise RuntimeError(
            f"No expect file exists for {os.path.basename(expected_file)} in {expected_file}; "
            "to accept the current output, re-run the failing test after setting the EXPECTTEST_ACCEPT "
            "env variable. For example: EXPECTTEST_ACCEPT=1 pytest test/test_models.py -k alexnet"
        )

    return expected_file


def _assert_expected(output, name, prec):
    """Test that a python value matches the recorded contents of a file
    based on a "check" name. The value must be
    pickable with `torch.save`. This file
    is placed in the 'expect' directory in the same directory
    as the test script. You can automatically update the recorded test
    output using an EXPECTTEST_ACCEPT=1 env variable.
    """
    expected_file = _get_expected_file(name)

    if ACCEPT:
        filename = {os.path.basename(expected_file)}
        print("Accepting updated output for {}:\n\n{}".format(filename, output))
        torch.save(output, expected_file)
        MAX_PICKLE_SIZE = 50 * 1000  # 50 KB
        binary_size = os.path.getsize(expected_file)
        if binary_size > MAX_PICKLE_SIZE:
            raise RuntimeError("The output for {}, is larger than 50kb".format(filename))
    else:
        expected = torch.load(expected_file)
        rtol = atol = prec
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol, check_dtype=False)


def _check_jit_scriptable(nn_module, args, unwrapper=None, skip=False):
    """Check that a nn.Module's results in TorchScript match eager and that it can be exported"""

    def assert_export_import_module(m, args):
        """Check that the results of a model are the same after saving and loading"""
        def get_export_import_copy(m):
            """Save and load a TorchScript model"""
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            imported = torch.jit.load(buffer)
            return imported

        m_import = get_export_import_copy(m)
        with freeze_rng_state():
            results = m(*args)
        with freeze_rng_state():
            results_from_imported = m_import(*args)
        tol = 3e-4
        try:
            torch.testing.assert_close(results, results_from_imported, atol=tol, rtol=tol)
        except torch.testing._asserts.UsageError:
            # custom check for the models that return named tuples:
            # we compare field by field while ignoring None as assert_close can't handle None
            for a, b in zip(results, results_from_imported):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=tol, rtol=tol)

    TEST_WITH_SLOW = os.getenv('PYTORCH_TEST_WITH_SLOW', '0') == '1'
    if not TEST_WITH_SLOW or skip:
        # TorchScript is not enabled, skip these tests
        msg = "The check_jit_scriptable test for {} was skipped. " \
              "This test checks if the module's results in TorchScript " \
              "match eager and that it can be exported. To run these " \
              "tests make sure you set the environment variable " \
              "PYTORCH_TEST_WITH_SLOW=1 and that the test is not " \
              "manually skipped.".format(nn_module.__class__.__name__)
        warnings.warn(msg, RuntimeWarning)
        return None

    sm = torch.jit.script(nn_module)

    with freeze_rng_state():
        eager_out = nn_module(*args)

    with freeze_rng_state():
        script_out = sm(*args)
        if unwrapper:
            script_out = unwrapper(script_out)

    torch.testing.assert_close(eager_out, script_out, atol=1e-4, rtol=1e-4)
    assert_export_import_module(sm, args)


def _check_fx_compatible(model, inputs):
    model_fx = torch.fx.symbolic_trace(model)
    out = model(inputs)
    out_fx = model_fx(inputs)
    torch.testing.assert_close(out, out_fx)


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


# The following contains configuration parameters for all models which are used by
# the _test_*_model methods.
_model_params = {
    'inception_v3': {
        'input_shape': (1, 3, 299, 299)
    },
    'retinanet_resnet50_fpn': {
        'num_classes': 20,
        'score_thresh': 0.01,
        'min_size': 224,
        'max_size': 224,
        'input_shape': (3, 224, 224),
    },
    'keypointrcnn_resnet50_fpn': {
        'num_classes': 2,
        'min_size': 224,
        'max_size': 224,
        'box_score_thresh': 0.15,
        'input_shape': (3, 224, 224),
    },
    'fasterrcnn_resnet50_fpn': {
        'num_classes': 20,
        'min_size': 224,
        'max_size': 224,
        'input_shape': (3, 224, 224),
    },
    'maskrcnn_resnet50_fpn': {
        'num_classes': 10,
        'min_size': 224,
        'max_size': 224,
        'input_shape': (3, 224, 224),
    },
    'fasterrcnn_mobilenet_v3_large_fpn': {
        'box_score_thresh': 0.02076,
    },
    'fasterrcnn_mobilenet_v3_large_320_fpn': {
        'box_score_thresh': 0.02076,
        'rpn_pre_nms_top_n_test': 1000,
        'rpn_post_nms_top_n_test': 1000,
    }
}


def _make_sliced_model(model, stop_layer):
    layers = OrderedDict()
    for name, layer in model.named_children():
        layers[name] = layer
        if name == stop_layer:
            break
    new_model = torch.nn.Sequential(layers)
    return new_model


@pytest.mark.parametrize('model_name', ['densenet121', 'densenet169', 'densenet201', 'densenet161'])
def test_memory_efficient_densenet(model_name):
    input_shape = (1, 3, 300, 300)
    x = torch.rand(input_shape)

    model1 = models.__dict__[model_name](num_classes=50, memory_efficient=True)
    params = model1.state_dict()
    num_params = sum([x.numel() for x in model1.parameters()])
    model1.eval()
    out1 = model1(x)
    out1.sum().backward()
    num_grad = sum([x.grad.numel() for x in model1.parameters() if x.grad is not None])

    model2 = models.__dict__[model_name](num_classes=50, memory_efficient=False)
    model2.load_state_dict(params)
    model2.eval()
    out2 = model2(x)

    assert num_params == num_grad
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize('dilate_layer_2', (True, False))
@pytest.mark.parametrize('dilate_layer_3', (True, False))
@pytest.mark.parametrize('dilate_layer_4', (True, False))
def test_resnet_dilation(dilate_layer_2, dilate_layer_3, dilate_layer_4):
    # TODO improve tests to also check that each layer has the right dimensionality
    model = models.__dict__["resnet50"](replace_stride_with_dilation=(dilate_layer_2, dilate_layer_3, dilate_layer_4))
    model = _make_sliced_model(model, stop_layer="layer4")
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    f = 2 ** sum((dilate_layer_2, dilate_layer_3, dilate_layer_4))
    assert out.shape == (1, 2048, 7 * f, 7 * f)


def test_mobilenet_v2_residual_setting():
    model = models.__dict__["mobilenet_v2"](inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    assert out.shape[-1] == 1000


@pytest.mark.parametrize('model_name', ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"])
def test_mobilenet_norm_layer(model_name):
    model = models.__dict__[model_name]()
    assert any(isinstance(x, nn.BatchNorm2d) for x in model.modules())

    def get_gn(num_channels):
        return nn.GroupNorm(32, num_channels)

    model = models.__dict__[model_name](norm_layer=get_gn)
    assert not(any(isinstance(x, nn.BatchNorm2d) for x in model.modules()))
    assert any(isinstance(x, nn.GroupNorm) for x in model.modules())


def test_inception_v3_eval():
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
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))


def test_fasterrcnn_double():
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
    model.double()
    model.eval()
    input_shape = (3, 300, 300)
    x = torch.rand(input_shape, dtype=torch.float64)
    model_input = [x]
    out = model(model_input)
    assert model_input[0] is x
    assert len(out) == 1
    assert "boxes" in out[0]
    assert "scores" in out[0]
    assert "labels" in out[0]


def test_googlenet_eval():
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
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))


@needs_cuda
def test_fasterrcnn_switch_devices():
    def checkOut(out):
        assert len(out) == 1
        assert "boxes" in out[0]
        assert "scores" in out[0]
        assert "labels" in out[0]

    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
    model.cuda()
    model.eval()
    input_shape = (3, 300, 300)
    x = torch.rand(input_shape, device='cuda')
    model_input = [x]
    out = model(model_input)
    assert model_input[0] is x

    checkOut(out)

    with torch.cuda.amp.autocast():
        out = model(model_input)

    checkOut(out)

    # now switch to cpu and make sure it works
    model.cpu()
    x = x.cpu()
    out_cpu = model([x])

    checkOut(out_cpu)


def test_generalizedrcnn_transform_repr():

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
    assert t.__repr__() == expected_string


@pytest.mark.parametrize('model_name', get_available_classification_models())
@pytest.mark.parametrize('dev', cpu_and_gpu())
def test_classification_model(model_name, dev):
    set_rng_seed(0)
    defaults = {
        'num_classes': 50,
        'input_shape': (1, 3, 224, 224),
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop('input_shape')

    model = models.__dict__[model_name](**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _assert_expected(out.cpu(), model_name, prec=0.1)
    assert out.shape[-1] == 50
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None))
    _check_fx_compatible(model, x)

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(x)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                _assert_expected(out.cpu(), model_name, prec=0.1)
            assert out.shape[-1] == 50


@pytest.mark.parametrize('model_name', get_available_segmentation_models())
@pytest.mark.parametrize('dev', cpu_and_gpu())
def test_segmentation_model(model_name, dev):
    set_rng_seed(0)
    defaults = {
        'num_classes': 10,
        'pretrained_backbone': False,
        'input_shape': (1, 3, 32, 32),
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop('input_shape')

    model = models.segmentation.__dict__[model_name](**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)["out"]

    def check_out(out):
        prec = 0.01
        try:
            # We first try to assert the entire output if possible. This is not
            # only the best way to assert results but also handles the cases
            # where we need to create a new expected result.
            _assert_expected(out.cpu(), model_name, prec=prec)
        except AssertionError:
            # Unfortunately some segmentation models are flaky with autocast
            # so instead of validating the probability scores, check that the class
            # predictions match.
            expected_file = _get_expected_file(model_name)
            expected = torch.load(expected_file)
            torch.testing.assert_close(out.argmax(dim=1), expected.argmax(dim=1), rtol=prec, atol=prec)
            return False  # Partial validation performed

        return True  # Full validation performed

    full_validation = check_out(out)

    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None))
    _check_fx_compatible(model, x)

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(x)["out"]
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                full_validation &= check_out(out)

    if not full_validation:
        msg = "The output of {} could only be partially validated. " \
              "This is likely due to unit-test flakiness, but you may " \
              "want to do additional manual checks if you made " \
              "significant changes to the codebase.".format(test_segmentation_model.__name__)
        warnings.warn(msg, RuntimeWarning)
        pytest.skip(msg)


@pytest.mark.parametrize('model_name', get_available_detection_models())
@pytest.mark.parametrize('dev', cpu_and_gpu())
def test_detection_model(model_name, dev):
    set_rng_seed(0)
    defaults = {
        'num_classes': 50,
        'pretrained_backbone': False,
        'input_shape': (3, 300, 300),
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop('input_shape')

    model = models.detection.__dict__[model_name](**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    model_input = [x]
    out = model(model_input)
    assert model_input[0] is x

    def check_out(out):
        assert len(out) == 1

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
            _assert_expected(output, model_name, prec=prec)
        except AssertionError:
            # Unfortunately detection models are flaky due to the unstable sort
            # in NMS. If matching across all outputs fails, use the same approach
            # as in NMSTester.test_nms_cuda to see if this is caused by duplicate
            # scores.
            expected_file = _get_expected_file(model_name)
            expected = torch.load(expected_file)
            torch.testing.assert_close(output[0]["scores"], expected[0]["scores"], rtol=prec, atol=prec,
                                       check_device=False, check_dtype=False)

            # Note: Fmassa proposed turning off NMS by adapting the threshold
            # and then using the Hungarian algorithm as in DETR to find the
            # best match between output and expected boxes and eliminate some
            # of the flakiness. Worth exploring.
            return False  # Partial validation performed

        return True  # Full validation performed

    full_validation = check_out(out)
    _check_jit_scriptable(model, ([x],), unwrapper=script_model_unwrapper.get(model_name, None))

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(model_input)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                full_validation &= check_out(out)

    if not full_validation:
        msg = "The output of {} could only be partially validated. " \
              "This is likely due to unit-test flakiness, but you may " \
              "want to do additional manual checks if you made " \
              "significant changes to the codebase.".format(test_detection_model.__name__)
        warnings.warn(msg, RuntimeWarning)
        pytest.skip(msg)


@pytest.mark.parametrize('model_name', get_available_detection_models())
def test_detection_model_validation(model_name):
    set_rng_seed(0)
    model = models.detection.__dict__[model_name](num_classes=50, pretrained_backbone=False)
    input_shape = (3, 300, 300)
    x = [torch.rand(input_shape)]

    # validate that targets are present in training
    with pytest.raises(ValueError):
        model(x)

    # validate type
    targets = [{'boxes': 0.}]
    with pytest.raises(ValueError):
        model(x, targets=targets)

    # validate boxes shape
    for boxes in (torch.rand((4,)), torch.rand((1, 5))):
        targets = [{'boxes': boxes}]
        with pytest.raises(ValueError):
            model(x, targets=targets)

    # validate that no degenerate boxes are present
    boxes = torch.tensor([[1, 3, 1, 4], [2, 4, 3, 4]])
    targets = [{'boxes': boxes}]
    with pytest.raises(ValueError):
        model(x, targets=targets)


@pytest.mark.parametrize('model_name', get_available_video_models())
@pytest.mark.parametrize('dev', cpu_and_gpu())
def test_video_model(model_name, dev):
    # the default input shape is
    # bs * num_channels * clip_len * h *w
    input_shape = (1, 3, 4, 112, 112)
    # test both basicblock and Bottleneck
    model = models.video.__dict__[model_name](num_classes=50)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None))
    _check_fx_compatible(model, x)
    assert out.shape[-1] == 50

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(x)
            assert out.shape[-1] == 50


@pytest.mark.skipif(not ('fbgemm' in torch.backends.quantized.supported_engines and
                         'qnnpack' in torch.backends.quantized.supported_engines),
                    reason="This Pytorch Build has not been built with fbgemm and qnnpack")
@pytest.mark.parametrize('model_name', get_available_quantizable_models())
def test_quantized_classification_model(model_name):
    defaults = {
        'input_shape': (1, 3, 224, 224),
        'pretrained': False,
        'quantize': True,
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop('input_shape')

    # First check if quantize=True provides models that can run with input data
    model = torchvision.models.quantization.__dict__[model_name](**kwargs)
    x = torch.rand(input_shape)
    model(x)

    kwargs['quantize'] = False
    for eval_mode in [True, False]:
        model = torchvision.models.quantization.__dict__[model_name](**kwargs)
        if eval_mode:
            model.eval()
            model.qconfig = torch.quantization.default_qconfig
        else:
            model.train()
            model.qconfig = torch.quantization.default_qat_qconfig

        model.fuse_model()
        if eval_mode:
            torch.quantization.prepare(model, inplace=True)
        else:
            torch.quantization.prepare_qat(model, inplace=True)
            model.eval()

        torch.quantization.convert(model, inplace=True)

    try:
        torch.jit.script(model)
    except Exception as e:
        tb = traceback.format_exc()
        raise AssertionError(f"model cannot be scripted. Traceback = {str(tb)}") from e


if __name__ == '__main__':
    pytest.main([__file__])
