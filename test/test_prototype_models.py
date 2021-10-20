import functools
import io
import operator
import os
import traceback
import warnings
from collections import OrderedDict

import pytest
import torch
import torch.fx
import torch.nn as nn
import torchvision
from _utils_internal import get_relative_path
from common_utils import map_nested_tensor_object, freeze_rng_state, set_rng_seed, cpu_and_gpu, needs_cuda
from torchvision.prototype import models
from test_models import _assert_expected, _model_params


model_to_default_weights_mapping = {
    "resnet18": ["ImageNet1K_RefV1", models.ResNet18Weights.ImageNet1K_RefV1],
    "resnet34": ["ImageNet1K_RefV1", models.ResNet34Weights.ImageNet1K_RefV1],
    "resnet50":["ImageNet1K_RefV1", models.ResNet50Weights.ImageNet1K_RefV1],
    "resnet101": ["ImageNet1K_RefV1", models.ResNet101Weights.ImageNet1K_RefV1],
    "resnet152": ["ImageNet1K_RefV1", models.ResNet152Weights.ImageNet1K_RefV1],
    "resnext50_32x4d": ["ImageNet1K_RefV1", models.ResNeXt50_32x4dWeights.ImageNet1K_RefV1],
    "resnext101_32x8d": ["ImageNet1K_RefV1", models.ResNeXt101_32x8dWeights.ImageNet1K_RefV1],
    "wide_resnet50_2": ["ImageNet1K_RefV1", models.WideResNet50_2Weights.ImageNet1K_RefV1],
    "wide_resnet101_2": ["ImageNet1K_RefV1", models.WideResNet101_2Weights.ImageNet1K_RefV1],
}

def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

@pytest.mark.parametrize("model_name", get_available_classification_models())
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "1", reason="Prototype code tests are disabled")
def test_classification_model(model_name, dev):
    set_rng_seed(0)
    defaults = {
        "num_classes": 50,
        "input_shape": (1, 3, 224, 224),
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    model = models.__dict__[model_name](**kwargs)
    model.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out = model(x)
    _assert_expected(out.cpu(), model_name, prec=0.1)
    assert out.shape[-1] == 50


@pytest.mark.parametrize("model_name", get_available_classification_models())
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "1") == "0", reason="Prototype code tests are disabled")
def test_old_vs_new_classification_factory(model_name, dev):
    set_rng_seed(0)
    defaults = {
        "num_classes": 50,
        "pretrained": True,
    }
    input_shape = (1, 3, 224, 224)
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    model_old = models.__dict__[model_name](**kwargs)
    model_old.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out_old = model_old(x)
    defaults.pop("pretrained")
    for weights_val in model_to_default_weights_mapping[model_name]:
        kwargs = {**defaults, **_model_params.get(model_name, {}), "weights": weights_val}
        model_new = models.__dict__[model_name](**kwargs)
        model_new.eval().to(device=dev)
        # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
        out_new = model_new(x)
        torch.testing.assert_close(out_new, out_old, rtol=0.0, atol=0.0, check_dtype=False)
