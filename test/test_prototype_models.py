import importlib
import os

import pytest
import test_models as TM
import torch
from common_utils import cpu_and_gpu
from torchvision.prototype import models


def _get_original_model(model_fn):
    original_module_name = model_fn.__module__.replace(".prototype", "")
    module = importlib.import_module(original_module_name)
    return module.__dict__[model_fn.__name__]


def _build_model(fn, **kwargs):
    try:
        model = fn(**kwargs)
    except ValueError as e:
        msg = str(e)
        if "No checkpoint is available" in msg:
            pytest.skip(msg)
        raise e
    return model.eval()


def get_models_with_module_names(module):
    module_name = module.__name__.split(".")[-1]
    return [(fn, module_name) for fn in TM.get_models_from_module(module)]


@pytest.mark.parametrize(
    "model_fn, weight",
    [
        (models.resnet50, models.ResNet50Weights.ImageNet1K_RefV2),
        (models.quantization.resnet50, models.quantization.QuantizedResNet50Weights.ImageNet1K_FBGEMM_RefV1),
    ],
)
def test_get_weight(model_fn, weight):
    assert models._api.get_weight(model_fn, weight.name) == weight


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_classification_model(model_fn, dev):
    TM.test_classification_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.quantization))
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_quantized_classification_model(model_fn):
    TM.test_quantized_classification_model(model_fn)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.segmentation))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_segmentation_model(model_fn, dev):
    TM.test_segmentation_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.video))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_video_model(model_fn, dev):
    TM.test_video_model(model_fn, dev)


@pytest.mark.parametrize(
    "model_fn, module_name",
    get_models_with_module_names(models)
    + get_models_with_module_names(models.quantization)
    + get_models_with_module_names(models.segmentation)
    + get_models_with_module_names(models.video),
)
@pytest.mark.parametrize("dev", cpu_and_gpu())
@pytest.mark.skipif(os.getenv("PYTORCH_TEST_WITH_PROTOTYPE", "0") == "0", reason="Prototype code tests are disabled")
def test_old_vs_new_factory(model_fn, module_name, dev):
    defaults = {
        "models": {
            "input_shape": (1, 3, 224, 224),
        },
        "quantization": {
            "input_shape": (1, 3, 224, 224),
        },
        "segmentation": {
            "input_shape": (1, 3, 520, 520),
        },
        "video": {
            "input_shape": (1, 3, 4, 112, 112),
        },
    }
    model_name = model_fn.__name__
    kwargs = {"pretrained": True, **defaults[module_name], **TM._model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    x = torch.rand(input_shape).to(device=dev)

    # compare with new model builder parameterized in the old fashion way
    model_old = _build_model(_get_original_model(model_fn), **kwargs).to(device=dev)
    model_new = _build_model(model_fn, **kwargs).to(device=dev)
    torch.testing.assert_close(model_new(x), model_old(x), rtol=0.0, atol=0.0, check_dtype=False)


def test_smoke():
    import torchvision.prototype.models  # noqa: F401
