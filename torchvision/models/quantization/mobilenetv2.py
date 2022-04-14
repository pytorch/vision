from functools import partial
from typing import Any, Optional, Union

from torch import Tensor
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2, MobileNet_V2_Weights

from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ImageClassification, InterpolationMode
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from .utils import _fuse_modules, _replace_relu, quantize_model


__all__ = [
    "QuantizableMobileNetV2",
    "MobileNet_V2_QuantizedWeights",
    "mobilenet_v2",
]


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                _fuse_modules(self.conv, [str(idx), str(idx + 1)], is_qat, inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                _fuse_modules(m, ["0", "1", "2"], is_qat, inplace=True)
            if type(m) is QuantizableInvertedResidual:
                m.fuse_model(is_qat)


class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "task": "image_classification",
            "architecture": "MobileNetV2",
            "publication_year": 2018,
            "num_params": 3504872,
            "size": (224, 224),
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "qnnpack",
            "quantization": "Quantization Aware Training",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNet_V2_Weights.IMAGENET1K_V1,
            "acc@1": 71.658,
            "acc@5": 90.150,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V2_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v2(
    *,
    weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
        weights (GoogLeNet_QuantizedWeights or GoogLeNet_Weights, optional): The pretrained
            weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize(bool): If True, returns a quantized model, else returns a float model
    """
    weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
