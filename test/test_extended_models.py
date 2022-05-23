import importlib
import os

import pytest
import test_models as TM
import torch
from torchvision import models
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._utils import handle_legacy_interface


run_if_test_with_extended = pytest.mark.skipif(
    os.getenv("PYTORCH_TEST_WITH_EXTENDED", "0") != "1",
    reason="Extended tests are disabled by default. Set PYTORCH_TEST_WITH_EXTENDED=1 to run them.",
)


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


@pytest.mark.parametrize(
    "name, weight",
    [
        ("ResNet50_Weights.IMAGENET1K_V1", models.ResNet50_Weights.IMAGENET1K_V1),
        ("ResNet50_Weights.DEFAULT", models.ResNet50_Weights.IMAGENET1K_V2),
        (
            "ResNet50_QuantizedWeights.DEFAULT",
            models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2,
        ),
        (
            "ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1",
            models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
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
    + TM.get_models_from_module(models.video)
    + TM.get_models_from_module(models.optical_flow),
)
def test_naming_conventions(model_fn):
    weights_enum = _get_model_weights(model_fn)
    assert weights_enum is not None
    assert len(weights_enum) == 0 or hasattr(weights_enum, "DEFAULT")


@pytest.mark.parametrize(
    "model_fn",
    TM.get_models_from_module(models)
    + TM.get_models_from_module(models.detection)
    + TM.get_models_from_module(models.quantization)
    + TM.get_models_from_module(models.segmentation)
    + TM.get_models_from_module(models.video)
    + TM.get_models_from_module(models.optical_flow),
)
@run_if_test_with_extended
def test_schema_meta_validation(model_fn):
    # list of all possible supported high-level fields for weights meta-data
    permitted_fields = {
        "backend",
        "categories",
        "keypoint_names",
        "license",
        "_metrics",
        "min_size",
        "num_params",
        "recipe",
        "unquantized",
        "_docs",
    }
    # mandatory fields for each computer vision task
    classification_fields = {"categories", ("_metrics", "ImageNet-1K", "acc@1"), ("_metrics", "ImageNet-1K", "acc@5")}
    defaults = {
        "all": {"_metrics", "min_size", "num_params", "recipe", "_docs"},
        "models": classification_fields,
        "detection": {"categories", ("_metrics", "COCO-val2017", "box_map")},
        "quantization": classification_fields | {"backend", "unquantized"},
        "segmentation": {
            "categories",
            ("_metrics", "COCO-val2017-VOC-labels", "miou"),
            ("_metrics", "COCO-val2017-VOC-labels", "pixel_acc"),
        },
        "video": {"categories", ("_metrics", "Kinetics-400", "acc@1"), ("_metrics", "Kinetics-400", "acc@5")},
        "optical_flow": set(),
    }
    model_name = model_fn.__name__
    module_name = model_fn.__module__.split(".")[-2]
    expected_fields = defaults["all"] | defaults[module_name]

    weights_enum = _get_model_weights(model_fn)
    if len(weights_enum) == 0:
        pytest.skip(f"Model '{model_name}' doesn't have any pre-trained weights.")

    problematic_weights = {}
    incorrect_params = []
    bad_names = []
    for w in weights_enum:
        actual_fields = set(w.meta.keys())
        actual_fields |= set(
            ("_metrics", dataset, metric_key)
            for dataset in w.meta.get("_metrics", {}).keys()
            for metric_key in w.meta.get("_metrics", {}).get(dataset, {}).keys()
        )
        missing_fields = expected_fields - actual_fields
        unsupported_fields = set(w.meta.keys()) - permitted_fields
        if missing_fields or unsupported_fields:
            problematic_weights[w] = {"missing": missing_fields, "unsupported": unsupported_fields}
        if w == weights_enum.DEFAULT:
            if module_name == "quantization":
                # parameters() count doesn't work well with quantization, so we check against the non-quantized
                unquantized_w = w.meta.get("unquantized")
                if unquantized_w is not None and w.meta.get("num_params") != unquantized_w.meta.get("num_params"):
                    incorrect_params.append(w)
            else:
                if w.meta.get("num_params") != sum(p.numel() for p in model_fn(weights=w).parameters()):
                    incorrect_params.append(w)
        else:
            if w.meta.get("num_params") != weights_enum.DEFAULT.meta.get("num_params"):
                if w.meta.get("num_params") != sum(p.numel() for p in model_fn(weights=w).parameters()):
                    incorrect_params.append(w)
        if not w.name.isupper():
            bad_names.append(w)

    assert not problematic_weights
    assert not incorrect_params
    assert not bad_names


@pytest.mark.parametrize(
    "model_fn",
    TM.get_models_from_module(models)
    + TM.get_models_from_module(models.detection)
    + TM.get_models_from_module(models.quantization)
    + TM.get_models_from_module(models.segmentation)
    + TM.get_models_from_module(models.video)
    + TM.get_models_from_module(models.optical_flow),
)
@run_if_test_with_extended
def test_transforms_jit(model_fn):
    model_name = model_fn.__name__
    weights_enum = _get_model_weights(model_fn)
    if len(weights_enum) == 0:
        pytest.skip(f"Model '{model_name}' doesn't have any pre-trained weights.")

    defaults = {
        "models": {
            "input_shape": (1, 3, 224, 224),
        },
        "detection": {
            "input_shape": (3, 300, 300),
        },
        "quantization": {
            "input_shape": (1, 3, 224, 224),
        },
        "segmentation": {
            "input_shape": (1, 3, 520, 520),
        },
        "video": {
            "input_shape": (1, 4, 3, 112, 112),
        },
        "optical_flow": {
            "input_shape": (1, 3, 128, 128),
        },
    }
    module_name = model_fn.__module__.split(".")[-2]

    kwargs = {**defaults[module_name], **TM._model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    x = torch.rand(input_shape)
    if module_name == "optical_flow":
        args = (x, x)
    else:
        args = (x,)

    problematic_weights = []
    for w in weights_enum:
        transforms = w.transforms()
        try:
            TM._check_jit_scriptable(transforms, args)
        except Exception:
            problematic_weights.append(w)

    assert not problematic_weights


# With this filter, every unexpected warning will be turned into an error
@pytest.mark.filterwarnings("error")
class TestHandleLegacyInterface:
    class ModelWeights(WeightsEnum):
        Sentinel = Weights(url="https://pytorch.org", transforms=lambda x: x, meta=dict())

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(), id="empty"),
            pytest.param(dict(weights=None), id="None"),
            pytest.param(dict(weights=ModelWeights.Sentinel), id="Weights"),
        ],
    )
    def test_no_warn(self, kwargs):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        builder(**kwargs)

    @pytest.mark.parametrize("pretrained", (True, False))
    def test_pretrained_pos(self, pretrained):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        with pytest.warns(UserWarning, match="positional"):
            builder(pretrained)

    @pytest.mark.parametrize("pretrained", (True, False))
    def test_pretrained_kw(self, pretrained):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        with pytest.warns(UserWarning, match="deprecated"):
            builder(pretrained)

    @pytest.mark.parametrize("pretrained", (True, False))
    @pytest.mark.parametrize("positional", (True, False))
    def test_equivalent_behavior_weights(self, pretrained, positional):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        args, kwargs = ((pretrained,), dict()) if positional else ((), dict(pretrained=pretrained))
        with pytest.warns(UserWarning, match=f"weights={self.ModelWeights.Sentinel if pretrained else None}"):
            builder(*args, **kwargs)

    def test_multi_params(self):
        weights_params = ("weights", "weights_other")
        pretrained_params = [param.replace("weights", "pretrained") for param in weights_params]

        @handle_legacy_interface(
            **{
                weights_param: (pretrained_param, self.ModelWeights.Sentinel)
                for weights_param, pretrained_param in zip(weights_params, pretrained_params)
            }
        )
        def builder(*, weights=None, weights_other=None):
            pass

        for pretrained_param in pretrained_params:
            with pytest.warns(UserWarning, match="deprecated"):
                builder(**{pretrained_param: True})

    def test_default_callable(self):
        @handle_legacy_interface(
            weights=(
                "pretrained",
                lambda kwargs: self.ModelWeights.Sentinel if kwargs["flag"] else None,
            )
        )
        def builder(*, weights=None, flag):
            pass

        with pytest.warns(UserWarning, match="deprecated"):
            builder(pretrained=True, flag=True)

        with pytest.raises(ValueError, match="weights"):
            builder(pretrained=True, flag=False)
