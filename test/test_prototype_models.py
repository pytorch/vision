import importlib
import os

import pytest
import torch
import test_models as TM
from common_utils import cpu_and_gpu
from torchvision.prototype import models


def _get_original_model(model_fn):
    original_module_name = model_fn.__module__.replace(".prototype", "")
    module = importlib.import_module(original_module_name)
    return module.__dict__[model_fn.__name__]


def test_get_weight():
    fn = models.resnet50
    weight_name = "ImageNet1K_RefV2"
    assert models._api.get_weight(fn, weight_name) == models.ResNet50Weights.ImageNet1K_RefV2


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_classification_model(model_fn, dev):
    TM.test_classification_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_old_vs_new_classification_factory(model_fn, dev):
    defaults = {
        "pretrained": True,
        "input_shape": (1, 3, 224, 224),
    }
    model_name = model_fn.__name__
    kwargs = {**defaults, **TM._model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    original_model_fn = _get_original_model(model_fn)
    model_old = original_model_fn(**kwargs)
    model_old.eval().to(device=dev)
    x = torch.rand(input_shape).to(device=dev)
    out_old = model_old(x)
    # compare with new model builder parameterized in the old fashion way
    model_new = model_fn(**kwargs)
    model_new.eval().to(device=dev)
    out_new = model_new(x)
    torch.testing.assert_close(out_new, out_old, rtol=0.0, atol=0.0, check_dtype=False)


def test_smoke():
    import torchvision.prototype.models  # noqa: F401
