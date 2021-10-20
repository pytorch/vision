# tests
import os

import pytest
import torch
import torch.fx
from common_utils import set_rng_seed, cpu_and_gpu
from test_models import _assert_expected, _model_params
from torchvision import models as original_models
from torchvision.prototype import models


def get_available_classification_models():
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


@pytest.mark.parametrize("model_name", get_available_classification_models())
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
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
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_old_vs_new_classification_factory(model_name, dev):
    set_rng_seed(0)
    defaults = {
        "pretrained": True,
        "input_shape": (1, 3, 224, 224),
    }
    kwargs = {**defaults, **_model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    model_old = original_models.__dict__[model_name](**kwargs)
    model_old.eval().to(device=dev)
    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
    x = torch.rand(input_shape).to(device=dev)
    out_old = model_old(x)
    # comapre with new model builder parameterized in the old fashion way
    model_new = models.__dict__[model_name](**kwargs)
    model_new.eval().to(device=dev)
    out_new = model_new(x)
    torch.testing.assert_close(out_new, out_old, rtol=0.0, atol=0.0, check_dtype=False)
