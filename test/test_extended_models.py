import copy
import os
import pickle

import pytest
import test_models as TM
import torch
from common_extended_utils import get_file_size_mb, get_ops
from torchvision import models
from torchvision.models import get_model_weights, Weights, WeightsEnum
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.detection.backbone_utils import mobilenet_backbone, resnet_fpn_backbone

run_if_test_with_extended = pytest.mark.skipif(
    os.getenv("PYTORCH_TEST_WITH_EXTENDED", "0") != "1",
    reason="Extended tests are disabled by default. Set PYTORCH_TEST_WITH_EXTENDED=1 to run them.",
)


@pytest.mark.parametrize(
    "name, model_class",
    [
        ("resnet50", models.ResNet),
        ("retinanet_resnet50_fpn_v2", models.detection.RetinaNet),
        ("raft_large", models.optical_flow.RAFT),
        ("quantized_resnet50", models.quantization.QuantizableResNet),
        ("lraspp_mobilenet_v3_large", models.segmentation.LRASPP),
        ("mvit_v1_b", models.video.MViT),
    ],
)
def test_get_model(name, model_class):
    assert isinstance(models.get_model(name), model_class)


@pytest.mark.parametrize(
    "name, model_fn",
    [
        ("resnet50", models.resnet50),
        ("retinanet_resnet50_fpn_v2", models.detection.retinanet_resnet50_fpn_v2),
        ("raft_large", models.optical_flow.raft_large),
        ("quantized_resnet50", models.quantization.resnet50),
        ("lraspp_mobilenet_v3_large", models.segmentation.lraspp_mobilenet_v3_large),
        ("mvit_v1_b", models.video.mvit_v1_b),
    ],
)
def test_get_model_builder(name, model_fn):
    assert models.get_model_builder(name) == model_fn


@pytest.mark.parametrize(
    "name, weight",
    [
        ("resnet50", models.ResNet50_Weights),
        ("retinanet_resnet50_fpn_v2", models.detection.RetinaNet_ResNet50_FPN_V2_Weights),
        ("raft_large", models.optical_flow.Raft_Large_Weights),
        ("quantized_resnet50", models.quantization.ResNet50_QuantizedWeights),
        ("lraspp_mobilenet_v3_large", models.segmentation.LRASPP_MobileNet_V3_Large_Weights),
        ("mvit_v1_b", models.video.MViT_V1_B_Weights),
    ],
)
def test_get_model_weights(name, weight):
    assert models.get_model_weights(name) == weight


@pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
@pytest.mark.parametrize(
    "name",
    [
        "resnet50",
        "retinanet_resnet50_fpn_v2",
        "raft_large",
        "quantized_resnet50",
        "lraspp_mobilenet_v3_large",
        "mvit_v1_b",
    ],
)
def test_weights_copyable(copy_fn, name):
    for weights in list(models.get_model_weights(name)):
        # It is somewhat surprising that (deep-)copying is an identity operation here, but this is the default behavior
        # of enums: https://docs.python.org/3/howto/enum.html#enum-members-aka-instances
        # Checking for equality, i.e. `==`, is sufficient (and even preferable) for our use case, should we need to drop
        # support for the identity operation in the future.
        assert copy_fn(weights) is weights


@pytest.mark.parametrize(
    "name",
    [
        "resnet50",
        "retinanet_resnet50_fpn_v2",
        "raft_large",
        "quantized_resnet50",
        "lraspp_mobilenet_v3_large",
        "mvit_v1_b",
    ],
)
def test_weights_deserializable(name):
    for weights in list(models.get_model_weights(name)):
        # It is somewhat surprising that deserialization is an identity operation here, but this is the default behavior
        # of enums: https://docs.python.org/3/howto/enum.html#enum-members-aka-instances
        # Checking for equality, i.e. `==`, is sufficient (and even preferable) for our use case, should we need to drop
        # support for the identity operation in the future.
        assert pickle.loads(pickle.dumps(weights)) is weights


def get_models_from_module(module):
    return [
        v.__name__
        for k, v in module.__dict__.items()
        if callable(v) and k[0].islower() and k[0] != "_" and k not in models._api.__all__
    ]


@pytest.mark.parametrize(
    "module", [models, models.detection, models.quantization, models.segmentation, models.video, models.optical_flow]
)
def test_list_models(module):
    a = set(get_models_from_module(module))
    b = set(x.replace("quantized_", "") for x in models.list_models(module))

    assert len(b) > 0
    assert a == b


@pytest.mark.parametrize(
    "include_filters",
    [
        None,
        [],
        (),
        "",
        "*resnet*",
        ["*alexnet*"],
        "*not-existing-model-for-test?",
        ["*resnet*", "*alexnet*"],
        ["*resnet*", "*alexnet*", "*not-existing-model-for-test?"],
        ("*resnet*", "*alexnet*"),
        set(["*resnet*", "*alexnet*"]),
    ],
)
@pytest.mark.parametrize(
    "exclude_filters",
    [
        None,
        [],
        (),
        "",
        "*resnet*",
        ["*alexnet*"],
        ["*not-existing-model-for-test?"],
        ["resnet34", "*not-existing-model-for-test?"],
        ["resnet34", "*resnet1*"],
        ("resnet34", "*resnet1*"),
        set(["resnet34", "*resnet1*"]),
    ],
)
def test_list_models_filters(include_filters, exclude_filters):
    actual = set(models.list_models(models, include=include_filters, exclude=exclude_filters))
    classification_models = set(get_models_from_module(models))

    if isinstance(include_filters, str):
        include_filters = [include_filters]
    if isinstance(exclude_filters, str):
        exclude_filters = [exclude_filters]

    if include_filters:
        expected = set()
        for include_f in include_filters:
            include_f = include_f.strip("*?")
            expected = expected | set(x for x in classification_models if include_f in x)
    else:
        expected = classification_models

    if exclude_filters:
        for exclude_f in exclude_filters:
            exclude_f = exclude_f.strip("*?")
            if exclude_f != "":
                a_exclude = set(x for x in classification_models if exclude_f in x)
                expected = expected - a_exclude

    assert expected == actual


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
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
def test_naming_conventions(model_fn):
    weights_enum = get_model_weights(model_fn)
    assert weights_enum is not None
    assert len(weights_enum) == 0 or hasattr(weights_enum, "DEFAULT")


detection_models_input_dims = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": (320, 320),
    "fasterrcnn_mobilenet_v3_large_fpn": (800, 800),
    "fasterrcnn_resnet50_fpn": (800, 800),
    "fasterrcnn_resnet50_fpn_v2": (800, 800),
    "fcos_resnet50_fpn": (800, 800),
    "keypointrcnn_resnet50_fpn": (1333, 1333),
    "maskrcnn_resnet50_fpn": (800, 800),
    "maskrcnn_resnet50_fpn_v2": (800, 800),
    "retinanet_resnet50_fpn": (800, 800),
    "retinanet_resnet50_fpn_v2": (800, 800),
    "ssd300_vgg16": (300, 300),
    "ssdlite320_mobilenet_v3_large": (320, 320),
}


@pytest.mark.parametrize(
    "model_fn",
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
@run_if_test_with_extended
def test_schema_meta_validation(model_fn):
    if model_fn.__name__ == "maskrcnn_resnet50_fpn_v2":
        pytest.skip(reason="FIXME https://github.com/pytorch/vision/issues/7349")

    # list of all possible supported high-level fields for weights meta-data
    permitted_fields = {
        "backend",
        "categories",
        "keypoint_names",
        "license",
        "_metrics",
        "min_size",
        "min_temporal_size",
        "num_params",
        "recipe",
        "unquantized",
        "_docs",
        "_ops",
        "_file_size",
    }
    # mandatory fields for each computer vision task
    classification_fields = {"categories", ("_metrics", "ImageNet-1K", "acc@1"), ("_metrics", "ImageNet-1K", "acc@5")}
    defaults = {
        "all": {"_metrics", "min_size", "num_params", "recipe", "_docs", "_file_size", "_ops"},
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

    weights_enum = get_model_weights(model_fn)
    if len(weights_enum) == 0:
        pytest.skip(f"Model '{model_name}' doesn't have any pre-trained weights.")

    problematic_weights = {}
    incorrect_meta = []
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

        if w == weights_enum.DEFAULT or any(w.meta[k] != weights_enum.DEFAULT.meta[k] for k in ["num_params", "_ops"]):
            if module_name == "quantization":
                # parameters() count doesn't work well with quantization, so we check against the non-quantized
                unquantized_w = w.meta.get("unquantized")
                if unquantized_w is not None:
                    if w.meta.get("num_params") != unquantized_w.meta.get("num_params"):
                        incorrect_meta.append((w, "num_params"))

                    # the methodology for quantized ops count doesn't work as well, so we take unquantized FLOPs
                    # instead
                    if w.meta["_ops"] != unquantized_w.meta.get("_ops"):
                        incorrect_meta.append((w, "_ops"))

            else:
                # loading the model and using it for parameter and ops verification
                model = model_fn(weights=w)

                if w.meta.get("num_params") != sum(p.numel() for p in model.parameters()):
                    incorrect_meta.append((w, "num_params"))

                kwargs = {}
                if model_name in detection_models_input_dims:
                    # detection models have non default height and width
                    height, width = detection_models_input_dims[model_name]
                    kwargs = {"height": height, "width": width}

                if not model_fn.__name__.startswith("vit"):
                    # FIXME: https://github.com/pytorch/vision/issues/7871
                    calculated_ops = get_ops(model=model, weight=w, **kwargs)
                    if calculated_ops != w.meta["_ops"]:
                        incorrect_meta.append((w, "_ops"))

        if not w.name.isupper():
            bad_names.append(w)

        if get_file_size_mb(w) != w.meta.get("_file_size"):
            incorrect_meta.append((w, "_file_size"))

    assert not problematic_weights
    assert not incorrect_meta
    assert not bad_names


@pytest.mark.parametrize(
    "model_fn",
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
@run_if_test_with_extended
def test_transforms_jit(model_fn):
    model_name = model_fn.__name__
    weights_enum = get_model_weights(model_fn)
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
            "input_shape": (1, 3, 4, 112, 112),
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
        if module_name == "video":
            x = x.permute(0, 2, 1, 3, 4)
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

    @pytest.mark.parametrize(
        "model_fn",
        [fn for fn in TM.list_model_fns(models) if fn.__name__ not in {"vit_h_14", "regnet_y_128gf"}]
        + TM.list_model_fns(models.detection)
        + TM.list_model_fns(models.quantization)
        + TM.list_model_fns(models.segmentation)
        + TM.list_model_fns(models.video)
        + TM.list_model_fns(models.optical_flow)
        + [
            lambda pretrained: resnet_fpn_backbone(backbone_name="resnet50", pretrained=pretrained),
            lambda pretrained: mobilenet_backbone(backbone_name="mobilenet_v2", fpn=False, pretrained=pretrained),
        ],
    )
    @run_if_test_with_extended
    def test_pretrained_deprecation(self, model_fn):
        with pytest.warns(UserWarning, match="deprecated"):
            model_fn(pretrained=True)
