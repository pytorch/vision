import importlib

import pytest
import test_models as TM
import torch
from common_utils import cpu_and_gpu, run_on_env_var
from torchvision.prototype import models

run_if_test_with_prototype = run_on_env_var(
    "PYTORCH_TEST_WITH_PROTOTYPE",
    skip_reason="Prototype tests are disabled by default. Set PYTORCH_TEST_WITH_PROTOTYPE=1 to run them.",
)


def _get_original_model(model_fn):
    original_module_name = model_fn.__module__.replace(".prototype", "")
    module = importlib.import_module(original_module_name)
    return module.__dict__[model_fn.__name__]


def _get_parent_module(model_fn):
    parent_module_name = ".".join(model_fn.__module__.split(".")[:-1])
    module = importlib.import_module(parent_module_name)
    return module


def _get_model_weights(model_fn):
    module = _get_parent_module(model_fn)
    weights_name = "_QuantizedWeights" if module.__name__.split(".")[-1] == "quantization" else "_Weights"
    try:
        return next(
            v
            for k, v in module.__dict__.items()
            if k.endswith(weights_name) and k.replace(weights_name, "").lower() == model_fn.__name__
        )
    except StopIteration:
        return None


def _build_model(fn, **kwargs):
    try:
        model = fn(**kwargs)
    except ValueError as e:
        msg = str(e)
        if "No checkpoint is available" in msg:
            pytest.skip(msg)
        raise e
    return model.eval()


@pytest.mark.parametrize(
    "name, weight",
    [
        ("ResNet50_Weights.ImageNet1K_V1", models.ResNet50_Weights.ImageNet1K_V1),
        ("ResNet50_Weights.default", models.ResNet50_Weights.ImageNet1K_V2),
        (
            "ResNet50_QuantizedWeights.default",
            models.quantization.ResNet50_QuantizedWeights.ImageNet1K_FBGEMM_V2,
        ),
        (
            "ResNet50_QuantizedWeights.ImageNet1K_FBGEMM_V1",
            models.quantization.ResNet50_QuantizedWeights.ImageNet1K_FBGEMM_V1,
        ),
    ],
)
def test_get_weight(name, weight):
    assert models.get_weight(name) == weight


@pytest.mark.parametrize(
    "model_fn",
    TM.get_models_from_module(models)
    + TM.get_models_from_module(models.detection)
    + TM.get_models_from_module(models.quantization)
    + TM.get_models_from_module(models.segmentation)
    + TM.get_models_from_module(models.video),
)
def test_naming_conventions(model_fn):
    weights_enum = _get_model_weights(model_fn)
    assert weights_enum is not None
    assert len(weights_enum) == 0 or hasattr(weights_enum, "default")


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@run_if_test_with_prototype
def test_classification_model(model_fn, dev):
    TM.test_classification_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.detection))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@run_if_test_with_prototype
def test_detection_model(model_fn, dev):
    TM.test_detection_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.quantization))
@run_if_test_with_prototype
def test_quantized_classification_model(model_fn):
    TM.test_quantized_classification_model(model_fn)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.segmentation))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@run_if_test_with_prototype
def test_segmentation_model(model_fn, dev):
    TM.test_segmentation_model(model_fn, dev)


@pytest.mark.parametrize("model_fn", TM.get_models_from_module(models.video))
@pytest.mark.parametrize("dev", cpu_and_gpu())
@run_if_test_with_prototype
def test_video_model(model_fn, dev):
    TM.test_video_model(model_fn, dev)


@pytest.mark.parametrize(
    "model_fn",
    TM.get_models_from_module(models)
    + TM.get_models_from_module(models.detection)
    + TM.get_models_from_module(models.quantization)
    + TM.get_models_from_module(models.segmentation)
    + TM.get_models_from_module(models.video),
)
@pytest.mark.parametrize("dev", cpu_and_gpu())
@run_if_test_with_prototype
def test_old_vs_new_factory(model_fn, dev):
    defaults = {
        "models": {
            "input_shape": (1, 3, 224, 224),
        },
        "detection": {
            "input_shape": (3, 300, 300),
        },
        "quantization": {
            "input_shape": (1, 3, 224, 224),
            "quantize": True,
        },
        "segmentation": {
            "input_shape": (1, 3, 520, 520),
        },
        "video": {
            "input_shape": (1, 3, 4, 112, 112),
        },
    }
    model_name = model_fn.__name__
    module_name = model_fn.__module__.split(".")[-2]
    kwargs = {"pretrained": True, **defaults[module_name], **TM._model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    kwargs.pop("num_classes", None)  # ignore this as it's an incompatible speed optimization for pre-trained models
    x = torch.rand(input_shape).to(device=dev)
    if module_name == "detection":
        x = [x]

    # compare with new model builder parameterized in the old fashion way
    try:
        model_old = _build_model(_get_original_model(model_fn), **kwargs).to(device=dev)
        model_new = _build_model(model_fn, **kwargs).to(device=dev)
    except ModuleNotFoundError:
        pytest.skip(f"Model '{model_name}' not available in both modules.")
    torch.testing.assert_close(model_new(x), model_old(x), rtol=0.0, atol=0.0, check_dtype=False)


def test_smoke():
    import torchvision.prototype.models  # noqa: F401
