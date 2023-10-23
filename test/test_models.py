import contextlib
import functools
import operator
import os
import pkgutil
import platform
import sys
import warnings
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
import torch.fx
import torch.nn as nn
from _utils_internal import get_relative_path
from common_utils import cpu_and_cuda, freeze_rng_state, map_nested_tensor_object, needs_cuda, set_rng_seed
from PIL import Image
from torchvision import models, transforms
from torchvision.models import get_model_builder, list_models


ACCEPT = os.getenv("EXPECTTEST_ACCEPT", "0") == "1"
SKIP_BIG_MODEL = os.getenv("SKIP_BIG_MODEL", "1") == "1"


def list_model_fns(module):
    return [get_model_builder(name) for name in list_models(module)]


def _get_image(input_shape, real_image, device, dtype=None):
    """This routine loads a real or random image based on `real_image` argument.
    Currently, the real image is utilized for the following list of models:
    - `retinanet_resnet50_fpn`,
    - `retinanet_resnet50_fpn_v2`,
    - `keypointrcnn_resnet50_fpn`,
    - `fasterrcnn_resnet50_fpn`,
    - `fasterrcnn_resnet50_fpn_v2`,
    - `fcos_resnet50_fpn`,
    - `maskrcnn_resnet50_fpn`,
    - `maskrcnn_resnet50_fpn_v2`,
    in `test_classification_model` and `test_detection_model`.
    To do so, a keyword argument `real_image` was added to the abovelisted models in `_model_params`
    """
    if real_image:
        # TODO: Maybe unify file discovery logic with test_image.py
        GRACE_HOPPER = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
        )

        img = Image.open(GRACE_HOPPER)

        original_width, original_height = img.size

        # make the image square
        img = img.crop((0, 0, original_width, original_width))
        img = img.resize(input_shape[1:3])

        convert_tensor = transforms.ToTensor()
        image = convert_tensor(img)
        assert tuple(image.size()) == input_shape
        return image.to(device=device, dtype=dtype)

    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    return torch.rand(input_shape).to(device=device, dtype=dtype)


@pytest.fixture
def disable_weight_loading(mocker):
    """When testing models, the two slowest operations are the downloading of the weights to a file and loading them
    into the model. Unless, you want to test against specific weights, these steps can be disabled without any
    drawbacks.

    Including this fixture into the signature of your test, i.e. `test_foo(disable_weight_loading)`, will recurse
    through all models in `torchvision.models` and will patch all occurrences of the function
    `download_state_dict_from_url` as well as the method `load_state_dict` on all subclasses of `nn.Module` to be
    no-ops.

    .. warning:

        Loaded models are still executable as normal, but will always have random weights. Make sure to not use this
        fixture if you want to compare the model output against reference values.

    """
    starting_point = models
    function_name = "load_state_dict_from_url"
    method_name = "load_state_dict"

    module_names = {info.name for info in pkgutil.walk_packages(starting_point.__path__, f"{starting_point.__name__}.")}
    targets = {f"torchvision._internally_replaced_utils.{function_name}", f"torch.nn.Module.{method_name}"}
    for name in module_names:
        module = sys.modules.get(name)
        if not module:
            continue

        if function_name in module.__dict__:
            targets.add(f"{module.__name__}.{function_name}")

        targets.update(
            {
                f"{module.__name__}.{obj.__name__}.{method_name}"
                for obj in module.__dict__.values()
                if isinstance(obj, type) and issubclass(obj, nn.Module) and method_name in obj.__dict__
            }
        )

    for target in targets:
        # See https://github.com/pytorch/vision/pull/4867#discussion_r743677802 for details
        with contextlib.suppress(AttributeError):
            mocker.patch(target)


def _get_expected_file(name=None):
    # Determine expected file based on environment
    expected_file_base = get_relative_path(os.path.realpath(__file__), "expect")

    # Note: for legacy reasons, the reference file names all had "ModelTest.test_" in their names
    # We hardcode it here to avoid having to re-generate the reference files
    expected_file = os.path.join(expected_file_base, "ModelTester.test_" + name)
    expected_file += "_expect.pkl"

    if not ACCEPT and not os.path.exists(expected_file):
        raise RuntimeError(
            f"No expect file exists for {os.path.basename(expected_file)} in {expected_file}; "
            "to accept the current output, re-run the failing test after setting the EXPECTTEST_ACCEPT "
            "env variable. For example: EXPECTTEST_ACCEPT=1 pytest test/test_models.py -k alexnet"
        )

    return expected_file


def _assert_expected(output, name, prec=None, atol=None, rtol=None):
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
        print(f"Accepting updated output for {filename}:\n\n{output}")
        torch.save(output, expected_file)
        MAX_PICKLE_SIZE = 50 * 1000  # 50 KB
        binary_size = os.path.getsize(expected_file)
        if binary_size > MAX_PICKLE_SIZE:
            raise RuntimeError(f"The output for {filename}, is larger than 50kb - got {binary_size}kb")
    else:
        expected = torch.load(expected_file)
        rtol = rtol or prec  # keeping prec param for legacy reason, but could be removed ideally
        atol = atol or prec
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol, check_dtype=False, check_device=False)


def _check_jit_scriptable(nn_module, args, unwrapper=None, eager_out=None):
    """Check that a nn.Module's results in TorchScript match eager and that it can be exported"""

    def get_export_import_copy(m):
        """Save and load a TorchScript model"""
        with TemporaryDirectory() as dir:
            path = os.path.join(dir, "script.pt")
            m.save(path)
            imported = torch.jit.load(path)
        return imported

    sm = torch.jit.script(nn_module)
    sm.eval()

    if eager_out is None:
        with torch.no_grad(), freeze_rng_state():
            eager_out = nn_module(*args)

    with torch.no_grad(), freeze_rng_state():
        script_out = sm(*args)
        if unwrapper:
            script_out = unwrapper(script_out)

    torch.testing.assert_close(eager_out, script_out, atol=1e-4, rtol=1e-4)

    m_import = get_export_import_copy(sm)
    with torch.no_grad(), freeze_rng_state():
        imported_script_out = m_import(*args)
        if unwrapper:
            imported_script_out = unwrapper(imported_script_out)

    torch.testing.assert_close(script_out, imported_script_out, atol=3e-4, rtol=3e-4)


def _check_fx_compatible(model, inputs, eager_out=None):
    model_fx = torch.fx.symbolic_trace(model)
    if eager_out is None:
        eager_out = model(inputs)
    with torch.no_grad(), freeze_rng_state():
        fx_out = model_fx(inputs)
    torch.testing.assert_close(eager_out, fx_out)


def _check_input_backprop(model, inputs):
    if isinstance(inputs, list):
        requires_grad = list()
        for inp in inputs:
            requires_grad.append(inp.requires_grad)
            inp.requires_grad_(True)
    else:
        requires_grad = inputs.requires_grad
        inputs.requires_grad_(True)

    out = model(inputs)

    if isinstance(out, dict):
        out["out"].sum().backward()
    else:
        if isinstance(out[0], dict):
            out[0]["scores"].sum().backward()
        else:
            out[0].sum().backward()

    if isinstance(inputs, list):
        for i, inp in enumerate(inputs):
            assert inputs[i].grad is not None
            inp.requires_grad_(requires_grad[i])
    else:
        assert inputs.grad is not None
        inputs.requires_grad_(requires_grad)


# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_model_unwrapper = {
    "googlenet": lambda x: x.logits,
    "inception_v3": lambda x: x.logits,
    "fasterrcnn_resnet50_fpn": lambda x: x[1],
    "fasterrcnn_resnet50_fpn_v2": lambda x: x[1],
    "fasterrcnn_mobilenet_v3_large_fpn": lambda x: x[1],
    "fasterrcnn_mobilenet_v3_large_320_fpn": lambda x: x[1],
    "maskrcnn_resnet50_fpn": lambda x: x[1],
    "maskrcnn_resnet50_fpn_v2": lambda x: x[1],
    "keypointrcnn_resnet50_fpn": lambda x: x[1],
    "retinanet_resnet50_fpn": lambda x: x[1],
    "retinanet_resnet50_fpn_v2": lambda x: x[1],
    "ssd300_vgg16": lambda x: x[1],
    "ssdlite320_mobilenet_v3_large": lambda x: x[1],
    "fcos_resnet50_fpn": lambda x: x[1],
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
    "maskrcnn_resnet50_fpn_v2",
    "keypointrcnn_resnet50_fpn",
)

# The tests for the following quantized models are flaky possibly due to inconsistent
# rounding errors in different platforms. For this reason the input/output consistency
# tests under test_quantized_classification_model will be skipped for the following models.
quantized_flaky_models = ("inception_v3", "resnet50")

# The tests for the following detection models are flaky.
# We run those tests on float64 to avoid floating point errors.
# FIXME: we shouldn't have to do that :'/
detection_flaky_models = ("keypointrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2")


# The following contains configuration parameters for all models which are used by
# the _test_*_model methods.
_model_params = {
    "inception_v3": {"input_shape": (1, 3, 299, 299), "init_weights": True},
    "retinanet_resnet50_fpn": {
        "num_classes": 20,
        "score_thresh": 0.01,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "retinanet_resnet50_fpn_v2": {
        "num_classes": 20,
        "score_thresh": 0.01,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "keypointrcnn_resnet50_fpn": {
        "num_classes": 2,
        "min_size": 224,
        "max_size": 224,
        "box_score_thresh": 0.17,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_resnet50_fpn": {
        "num_classes": 20,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "num_classes": 20,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fcos_resnet50_fpn": {
        "num_classes": 2,
        "score_thresh": 0.05,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "maskrcnn_resnet50_fpn": {
        "num_classes": 10,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "maskrcnn_resnet50_fpn_v2": {
        "num_classes": 10,
        "min_size": 224,
        "max_size": 224,
        "input_shape": (3, 224, 224),
        "real_image": True,
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "box_score_thresh": 0.02076,
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "box_score_thresh": 0.02076,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
    },
    "vit_h_14": {
        "image_size": 56,
        "input_shape": (1, 3, 56, 56),
    },
    "mvit_v1_b": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "mvit_v2_s": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "s3d": {
        "input_shape": (1, 3, 16, 224, 224),
    },
    "googlenet": {"init_weights": True},
}
# speeding up slow models:
slow_models = [
    "convnext_base",
    "convnext_large",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet101_2",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_y_128gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]
for m in slow_models:
    _model_params[m] = {"input_shape": (1, 3, 64, 64)}


# skip big models to reduce memory usage on CI test. We can exclude combinations of (platform-system, device).
skipped_big_models = {
    "vit_h_14": {("Windows", "cpu"), ("Windows", "cuda")},
    "regnet_y_128gf": {("Windows", "cpu"), ("Windows", "cuda")},
    "mvit_v1_b": {("Windows", "cuda"), ("Linux", "cuda")},
    "mvit_v2_s": {("Windows", "cuda"), ("Linux", "cuda")},
}


def is_skippable(model_name, device):
    if model_name not in skipped_big_models:
        return False

    platform_system = platform.system()
    device_name = str(device).split(":")[0]

    return (platform_system, device_name) in skipped_big_models[model_name]


# The following contains configuration and expected values to be used tests that are model specific
_model_tests_values = {
    "retinanet_resnet50_fpn": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [36, 46, 65, 78, 88, 89],
    },
    "retinanet_resnet50_fpn_v2": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [44, 74, 131, 170, 200, 203],
    },
    "keypointrcnn_resnet50_fpn": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [48, 58, 77, 90, 100, 101],
    },
    "fasterrcnn_resnet50_fpn": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [30, 40, 59, 72, 82, 83],
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [50, 80, 137, 176, 206, 209],
    },
    "maskrcnn_resnet50_fpn": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [42, 52, 71, 84, 94, 95],
    },
    "maskrcnn_resnet50_fpn_v2": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [66, 96, 153, 192, 222, 225],
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "max_trainable": 6,
        "n_trn_params_per_layer": [22, 23, 44, 70, 91, 97, 100],
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "max_trainable": 6,
        "n_trn_params_per_layer": [22, 23, 44, 70, 91, 97, 100],
    },
    "ssd300_vgg16": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [45, 51, 57, 63, 67, 71],
    },
    "ssdlite320_mobilenet_v3_large": {
        "max_trainable": 6,
        "n_trn_params_per_layer": [96, 99, 138, 200, 239, 257, 266],
    },
    "fcos_resnet50_fpn": {
        "max_trainable": 5,
        "n_trn_params_per_layer": [54, 64, 83, 96, 106, 107],
    },
}


def _make_sliced_model(model, stop_layer):
    layers = OrderedDict()
    for name, layer in model.named_children():
        layers[name] = layer
        if name == stop_layer:
            break
    new_model = torch.nn.Sequential(layers)
    return new_model


@pytest.mark.parametrize("model_fn", [models.densenet121, models.densenet169, models.densenet201, models.densenet161])
def test_memory_efficient_densenet(model_fn):
    input_shape = (1, 3, 300, 300)
    x = torch.rand(input_shape)

    model1 = model_fn(num_classes=50, memory_efficient=True)
    params = model1.state_dict()
    num_params = sum(x.numel() for x in model1.parameters())
    model1.eval()
    out1 = model1(x)
    out1.sum().backward()
    num_grad = sum(x.grad.numel() for x in model1.parameters() if x.grad is not None)

    model2 = model_fn(num_classes=50, memory_efficient=False)
    model2.load_state_dict(params)
    model2.eval()
    out2 = model2(x)

    assert num_params == num_grad
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-5)

    _check_input_backprop(model1, x)
    _check_input_backprop(model2, x)


@pytest.mark.parametrize("dilate_layer_2", (True, False))
@pytest.mark.parametrize("dilate_layer_3", (True, False))
@pytest.mark.parametrize("dilate_layer_4", (True, False))
def test_resnet_dilation(dilate_layer_2, dilate_layer_3, dilate_layer_4):
    # TODO improve tests to also check that each layer has the right dimensionality
    model = models.resnet50(replace_stride_with_dilation=(dilate_layer_2, dilate_layer_3, dilate_layer_4))
    model = _make_sliced_model(model, stop_layer="layer4")
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    f = 2 ** sum((dilate_layer_2, dilate_layer_3, dilate_layer_4))
    assert out.shape == (1, 2048, 7 * f, 7 * f)


def test_mobilenet_v2_residual_setting():
    model = models.mobilenet_v2(inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    assert out.shape[-1] == 1000


@pytest.mark.parametrize("model_fn", [models.mobilenet_v2, models.mobilenet_v3_large, models.mobilenet_v3_small])
def test_mobilenet_norm_layer(model_fn):
    model = model_fn()
    assert any(isinstance(x, nn.BatchNorm2d) for x in model.modules())

    def get_gn(num_channels):
        return nn.GroupNorm(1, num_channels)

    model = model_fn(norm_layer=get_gn)
    assert not (any(isinstance(x, nn.BatchNorm2d) for x in model.modules()))
    assert any(isinstance(x, nn.GroupNorm) for x in model.modules())


def test_inception_v3_eval():
    kwargs = {}
    kwargs["transform_input"] = True
    kwargs["aux_logits"] = True
    kwargs["init_weights"] = False
    name = "inception_v3"
    model = models.Inception3(**kwargs)
    model.aux_logits = False
    model.AuxLogits = None
    model = model.eval()
    x = torch.rand(1, 3, 299, 299)
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))
    _check_input_backprop(model, x)


def test_fasterrcnn_double():
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, weights=None, weights_backbone=None)
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
    _check_input_backprop(model, model_input)


def test_googlenet_eval():
    kwargs = {}
    kwargs["transform_input"] = True
    kwargs["aux_logits"] = True
    kwargs["init_weights"] = False
    name = "googlenet"
    model = models.GoogLeNet(**kwargs)
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None
    model = model.eval()
    x = torch.rand(1, 3, 224, 224)
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(name, None))
    _check_input_backprop(model, x)


@needs_cuda
def test_fasterrcnn_switch_devices():
    def checkOut(out):
        assert len(out) == 1
        assert "boxes" in out[0]
        assert "scores" in out[0]
        assert "labels" in out[0]

    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, weights=None, weights_backbone=None)
    model.cuda()
    model.eval()
    input_shape = (3, 300, 300)
    x = torch.rand(input_shape, device="cuda")
    model_input = [x]
    out = model(model_input)
    assert model_input[0] is x

    checkOut(out)

    with torch.cuda.amp.autocast():
        out = model(model_input)

    checkOut(out)

    _check_input_backprop(model, model_input)

    # now switch to cpu and make sure it works
    model.cpu()
    x = x.cpu()
    out_cpu = model([x])

    checkOut(out_cpu)

    _check_input_backprop(model, [x])


def test_generalizedrcnn_transform_repr():

    min_size, max_size = 224, 299
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    t = models.detection.transform.GeneralizedRCNNTransform(
        min_size=min_size, max_size=max_size, image_mean=image_mean, image_std=image_std
    )

    # Check integrity of object __repr__ attribute
    expected_string = "GeneralizedRCNNTransform("
    _indent = "\n    "
    expected_string += f"{_indent}Normalize(mean={image_mean}, std={image_std})"
    expected_string += f"{_indent}Resize(min_size=({min_size},), max_size={max_size}, "
    expected_string += "mode='bilinear')\n)"
    assert t.__repr__() == expected_string


test_vit_conv_stem_configs = [
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=2, out_channels=64),
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=2, out_channels=128),
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=1, out_channels=128),
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=2, out_channels=256),
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=1, out_channels=256),
    models.vision_transformer.ConvStemConfig(kernel_size=3, stride=2, out_channels=512),
]


def vitc_b_16(**kwargs: Any):
    return models.VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        conv_stem_configs=test_vit_conv_stem_configs,
        **kwargs,
    )


@pytest.mark.parametrize("model_fn", [vitc_b_16])
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_vitc_models(model_fn, dev):
    test_classification_model(model_fn, dev)


@torch.backends.cudnn.flags(allow_tf32=False)  # see: https://github.com/pytorch/vision/issues/7618
@pytest.mark.parametrize("model_fn", list_model_fns(models))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_classification_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "num_classes": 50,
        "input_shape": (1, 3, 224, 224),
    }
    model_name = model_fn.__name__
    if SKIP_BIG_MODEL and is_skippable(model_name, dev):
        pytest.skip("Skipped to reduce memory usage. Set env var SKIP_BIG_MODEL=0 to enable test for this model")
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    num_classes = kwargs.get("num_classes")
    input_shape = kwargs.pop("input_shape")
    real_image = kwargs.pop("real_image", False)

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    x = _get_image(input_shape=input_shape, real_image=real_image, device=dev)
    out = model(x)
    # FIXME: this if/else is nasty and only here to please our CI prior to the
    # release. We rethink these tests altogether.
    if model_name == "resnet101":
        prec = 0.2
    else:
        # FIXME: this is probably still way too high.
        prec = 0.1
    _assert_expected(out.cpu(), model_name, prec=prec)
    assert out.shape[-1] == num_classes
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)

    if dev == "cuda":
        with torch.cuda.amp.autocast():
            out = model(x)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                _assert_expected(out.cpu(), model_name, prec=0.1)
            assert out.shape[-1] == 50

    _check_input_backprop(model, x)


@pytest.mark.parametrize("model_fn", list_model_fns(models.segmentation))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_segmentation_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "num_classes": 10,
        "weights_backbone": None,
        "input_shape": (1, 3, 32, 32),
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    with torch.no_grad(), freeze_rng_state():
        out = model(x)

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
            torch.testing.assert_close(
                out.argmax(dim=1), expected.argmax(dim=1), rtol=prec, atol=prec, check_device=False
            )
            return False  # Partial validation performed

        return True  # Full validation performed

    full_validation = check_out(out["out"])

    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)

    if dev == "cuda":
        with torch.cuda.amp.autocast(), torch.no_grad(), freeze_rng_state():
            out = model(x)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                full_validation &= check_out(out["out"])

    if not full_validation:
        msg = (
            f"The output of {test_segmentation_model.__name__} could only be partially validated. "
            "This is likely due to unit-test flakiness, but you may "
            "want to do additional manual checks if you made "
            "significant changes to the codebase."
        )
        warnings.warn(msg, RuntimeWarning)
        pytest.skip(msg)

    _check_input_backprop(model, x)


@pytest.mark.parametrize("model_fn", list_model_fns(models.detection))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_detection_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "num_classes": 50,
        "weights_backbone": None,
        "input_shape": (3, 300, 300),
    }
    model_name = model_fn.__name__
    if model_name in detection_flaky_models:
        dtype = torch.float64
    else:
        dtype = torch.get_default_dtype()
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    real_image = kwargs.pop("real_image", False)

    model = model_fn(**kwargs)
    model.eval().to(device=dev, dtype=dtype)
    x = _get_image(input_shape=input_shape, real_image=real_image, device=dev, dtype=dtype)
    model_input = [x]
    with torch.no_grad(), freeze_rng_state():
        out = model(model_input)
    assert model_input[0] is x

    def check_out(out):
        assert len(out) == 1

        def compact(tensor):
            tensor = tensor.cpu()
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
            return tensor[ith_index - 1 :: ith_index]

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
            torch.testing.assert_close(
                output[0]["scores"], expected[0]["scores"], rtol=prec, atol=prec, check_device=False, check_dtype=False
            )

            # Note: Fmassa proposed turning off NMS by adapting the threshold
            # and then using the Hungarian algorithm as in DETR to find the
            # best match between output and expected boxes and eliminate some
            # of the flakiness. Worth exploring.
            return False  # Partial validation performed

        return True  # Full validation performed

    full_validation = check_out(out)
    _check_jit_scriptable(model, ([x],), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)

    if dev == "cuda":
        with torch.cuda.amp.autocast(), torch.no_grad(), freeze_rng_state():
            out = model(model_input)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                full_validation &= check_out(out)

    if not full_validation:
        msg = (
            f"The output of {test_detection_model.__name__} could only be partially validated. "
            "This is likely due to unit-test flakiness, but you may "
            "want to do additional manual checks if you made "
            "significant changes to the codebase."
        )
        warnings.warn(msg, RuntimeWarning)
        pytest.skip(msg)

    _check_input_backprop(model, model_input)


@pytest.mark.parametrize("model_fn", list_model_fns(models.detection))
def test_detection_model_validation(model_fn):
    set_rng_seed(0)
    model = model_fn(num_classes=50, weights=None, weights_backbone=None)
    input_shape = (3, 300, 300)
    x = [torch.rand(input_shape)]

    # validate that targets are present in training
    with pytest.raises(AssertionError):
        model(x)

    # validate type
    targets = [{"boxes": 0.0}]
    with pytest.raises(AssertionError):
        model(x, targets=targets)

    # validate boxes shape
    for boxes in (torch.rand((4,)), torch.rand((1, 5))):
        targets = [{"boxes": boxes}]
        with pytest.raises(AssertionError):
            model(x, targets=targets)

    # validate that no degenerate boxes are present
    boxes = torch.tensor([[1, 3, 1, 4], [2, 4, 3, 4]])
    targets = [{"boxes": boxes}]
    with pytest.raises(AssertionError):
        model(x, targets=targets)


@pytest.mark.parametrize("model_fn", list_model_fns(models.video))
@pytest.mark.parametrize("dev", cpu_and_cuda())
def test_video_model(model_fn, dev):
    set_rng_seed(0)
    # the default input shape is
    # bs * num_channels * clip_len * h *w
    defaults = {
        "input_shape": (1, 3, 4, 112, 112),
        "num_classes": 50,
    }
    model_name = model_fn.__name__
    if SKIP_BIG_MODEL and is_skippable(model_name, dev):
        pytest.skip("Skipped to reduce memory usage. Set env var SKIP_BIG_MODEL=0 to enable test for this model")
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    num_classes = kwargs.get("num_classes")
    input_shape = kwargs.pop("input_shape")
    # test both basicblock and Bottleneck
    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _assert_expected(out.cpu(), model_name, prec=0.1)
    assert out.shape[-1] == num_classes
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)
    assert out.shape[-1] == num_classes

    if dev == "cuda":
        with torch.cuda.amp.autocast():
            out = model(x)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                _assert_expected(out.cpu(), model_name, prec=0.1)
            assert out.shape[-1] == num_classes

    _check_input_backprop(model, x)


@pytest.mark.skipif(
    not (
        "fbgemm" in torch.backends.quantized.supported_engines
        and "qnnpack" in torch.backends.quantized.supported_engines
    ),
    reason="This Pytorch Build has not been built with fbgemm and qnnpack",
)
@pytest.mark.parametrize("model_fn", list_model_fns(models.quantization))
def test_quantized_classification_model(model_fn):
    set_rng_seed(0)
    defaults = {
        "num_classes": 5,
        "input_shape": (1, 3, 224, 224),
        "quantize": True,
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    # First check if quantize=True provides models that can run with input data
    model = model_fn(**kwargs)
    model.eval()
    x = torch.rand(input_shape)
    out = model(x)

    if model_name not in quantized_flaky_models:
        _assert_expected(out.cpu(), model_name + "_quantized", prec=2e-2)
        assert out.shape[-1] == 5
        _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
        _check_fx_compatible(model, x, eager_out=out)
    else:
        try:
            torch.jit.script(model)
        except Exception as e:
            raise AssertionError("model cannot be scripted.") from e

    kwargs["quantize"] = False
    for eval_mode in [True, False]:
        model = model_fn(**kwargs)
        if eval_mode:
            model.eval()
            model.qconfig = torch.ao.quantization.default_qconfig
        else:
            model.train()
            model.qconfig = torch.ao.quantization.default_qat_qconfig

        model.fuse_model(is_qat=not eval_mode)
        if eval_mode:
            torch.ao.quantization.prepare(model, inplace=True)
        else:
            torch.ao.quantization.prepare_qat(model, inplace=True)
            model.eval()

        torch.ao.quantization.convert(model, inplace=True)


@pytest.mark.parametrize("model_fn", list_model_fns(models.detection))
def test_detection_model_trainable_backbone_layers(model_fn, disable_weight_loading):
    model_name = model_fn.__name__
    max_trainable = _model_tests_values[model_name]["max_trainable"]
    n_trainable_params = []
    for trainable_layers in range(0, max_trainable + 1):
        model = model_fn(weights=None, weights_backbone="DEFAULT", trainable_backbone_layers=trainable_layers)

        n_trainable_params.append(len([p for p in model.parameters() if p.requires_grad]))
    assert n_trainable_params == _model_tests_values[model_name]["n_trn_params_per_layer"]


@needs_cuda
@pytest.mark.parametrize("model_fn", list_model_fns(models.optical_flow))
@pytest.mark.parametrize("scripted", (False, True))
def test_raft(model_fn, scripted):

    torch.manual_seed(0)

    # We need very small images, otherwise the pickle size would exceed the 50KB
    # As a result we need to override the correlation pyramid to not downsample
    # too much, otherwise we would get nan values (effective H and W would be
    # reduced to 1)
    corr_block = models.optical_flow.raft.CorrBlock(num_levels=2, radius=2)

    model = model_fn(corr_block=corr_block).eval().to("cuda")
    if scripted:
        model = torch.jit.script(model)

    bs = 1
    img1 = torch.rand(bs, 3, 80, 72).cuda()
    img2 = torch.rand(bs, 3, 80, 72).cuda()

    preds = model(img1, img2)
    flow_pred = preds[-1]
    # Tolerance is fairly high, but there are 2 * H * W outputs to check
    # The .pkl were generated on the AWS cluter, on the CI it looks like the results are slightly different
    _assert_expected(flow_pred.cpu(), name=model_fn.__name__, atol=1e-2, rtol=1)


if __name__ == "__main__":
    pytest.main([__file__])
