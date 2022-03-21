import contextlib
import functools
import operator
import os
import pkgutil
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
from common_utils import map_nested_tensor_object, freeze_rng_state, set_rng_seed, cpu_and_gpu, needs_cuda
from torchvision import models

ACCEPT = os.getenv("EXPECTTEST_ACCEPT", "0") == "1"


def get_models_from_module(module):
    # TODO add a registration mechanism to torchvision.models
    return [
        v
        for k, v in module.__dict__.items()
        if callable(v) and k[0].lower() == k[0] and k[0] != "_" and k != "get_weight"
    ]


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
    else:
        expected = torch.load(expected_file)
        rtol = rtol or prec  # keeping prec param for legacy reason, but could be removed ideally
        atol = atol or prec
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol, check_dtype=False)


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

    if eager_out is None:
        with torch.no_grad(), freeze_rng_state():
            if unwrapper:
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
    "fasterrcnn_mobilenet_v3_large_fpn": lambda x: x[1],
    "fasterrcnn_mobilenet_v3_large_320_fpn": lambda x: x[1],
    "maskrcnn_resnet50_fpn": lambda x: x[1],
    "keypointrcnn_resnet50_fpn": lambda x: x[1],
    "retinanet_resnet50_fpn": lambda x: x[1],
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
)

# The tests for the following quantized models are flaky possibly due to inconsistent
# rounding errors in different platforms. For this reason the input/output consistency
# tests under test_quantized_classification_model will be skipped for the following models.
quantized_flaky_models = ("inception_v3", "resnet50")


# The following contains configuration parameters for all models which are used by
# the _test_*_model methods.
_model_params = {
    "inception_v3": {"input_shape": (1, 3, 299, 299)},
}
# speeding up slow models:
slow_models = [
    "convnext_base",
    "convnext_large",
    "resnext101_32x8d",
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
]
for m in slow_models:
    _model_params[m] = {"input_shape": (1, 3, 64, 64)}


@pytest.mark.parametrize("model_fn", get_models_from_module(models))
@pytest.mark.parametrize("dev", cpu_and_gpu())
def test_classification_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "input_shape": (1, 3, 224, 224),
        "pretrained": True,
    }
    model_name = model_fn.__name__
    if model_name in {"mnasnet0_75", "mnasnet1_3", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "regnet_y_128gf"}:
        return #  No checkpoints
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _assert_expected(out.cpu(), model_name, prec=0.1)
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(x)
            # See autocast_flaky_numerics comment at top of file.
            if model_name not in autocast_flaky_numerics:
                _assert_expected(out.cpu(), model_name, prec=0.1)

    _check_input_backprop(model, x)


@pytest.mark.parametrize("model_fn", get_models_from_module(models.segmentation))
@pytest.mark.parametrize("dev", cpu_and_gpu())
def test_segmentation_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "input_shape": (1, 3, 32, 32),
        "pretrained": True,
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
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
            torch.testing.assert_close(out.argmax(dim=1), expected.argmax(dim=1), rtol=prec, atol=prec)
            return False  # Partial validation performed

        return True  # Full validation performed

    full_validation = check_out(out["out"])

    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
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


@pytest.mark.parametrize("model_fn", get_models_from_module(models.detection))
@pytest.mark.parametrize("dev", cpu_and_gpu())
def test_detection_model(model_fn, dev):
    set_rng_seed(0)
    defaults = {
        "input_shape": (3, 300, 300),
        "pretrained": True,
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")

    model = model_fn(**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    model_input = [x]
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

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
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


@pytest.mark.parametrize("model_fn", get_models_from_module(models.video))
@pytest.mark.parametrize("dev", cpu_and_gpu())
def test_video_model(model_fn, dev):
    # the default input shape is
    # bs * num_channels * clip_len * h *w
    input_shape = (1, 3, 4, 112, 112)
    model_name = model_fn.__name__
    # test both basicblock and Bottleneck
    model = model_fn(pretrained=True)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _check_jit_scriptable(model, (x,), unwrapper=script_model_unwrapper.get(model_name, None), eager_out=out)
    _check_fx_compatible(model, x, eager_out=out)

    if dev == torch.device("cuda"):
        with torch.cuda.amp.autocast():
            out = model(x)

    _check_input_backprop(model, x)


@pytest.mark.skipif(
    not (
        "fbgemm" in torch.backends.quantized.supported_engines
        and "qnnpack" in torch.backends.quantized.supported_engines
    ),
    reason="This Pytorch Build has not been built with fbgemm and qnnpack",
)
@pytest.mark.parametrize("model_fn", get_models_from_module(models.quantization))
def test_quantized_classification_model(model_fn):
    set_rng_seed(0)
    defaults = {
        "input_shape": (1, 3, 224, 224),
        "pretrained": True,
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
        _assert_expected(out, model_name + "_quantized", prec=0.1)
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


@needs_cuda
@pytest.mark.parametrize("model_builder", (models.optical_flow.raft_large, models.optical_flow.raft_small))
@pytest.mark.parametrize("scripted", (False, True))
def test_raft(model_builder, scripted):

    torch.manual_seed(0)

    # We need very small images, otherwise the pickle size would exceed the 50KB
    # As a resut we need to override the correlation pyramid to not downsample
    # too much, otherwise we would get nan values (effective H and W would be
    # reduced to 1)
    corr_block = models.optical_flow.raft.CorrBlock(num_levels=2, radius=2)

    model = model_builder(corr_block=corr_block, pretrained=True).eval().to("cuda")
    if scripted:
        model = torch.jit.script(model)

    bs = 1
    img1 = torch.rand(bs, 3, 80, 72).cuda()
    img2 = torch.rand(bs, 3, 80, 72).cuda()

    preds = model(img1, img2)
    flow_pred = preds[-1]
    # Tolerance is fairly high, but there are 2 * H * W outputs to check
    # The .pkl were generated on the AWS cluter, on the CI it looks like the resuts are slightly different
    _assert_expected(flow_pred, name=model_builder.__name__, atol=1e-2, rtol=1)


if __name__ == "__main__":
    pytest.main([__file__])
