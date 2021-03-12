import torch
from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig, ConvBNActivation, MobileNetV3,\
    SqueezeExcitation, model_urls, _mobilenet_v3_conf
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from typing import Any, List, Optional
from .utils import _replace_relu


__all__ = ['QuantizableMobileNetV3', 'mobilenet_v3_large']

quant_model_urls = {
    'mobilenet_v3_large_qnnpack':
        "https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth",
}


class QuantizableSqueezeExcitation(SqueezeExcitation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input, False), input)

    def fuse_model(self):
        fuse_modules(self, ['fc1', 'relu'], inplace=True)


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, se_layer=QuantizableSqueezeExcitation, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.block(x))
        else:
            return self.block(x)


class QuantizableMobileNetV3(MobileNetV3):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNActivation:
                modules_to_fuse = ['0', '1']
                if type(m[2]) == nn.ReLU:
                    modules_to_fuse.append('2')
                fuse_modules(m, modules_to_fuse, inplace=True)
            elif type(m) == QuantizableSqueezeExcitation:
                m.fuse_model()


def _load_weights(
    arch: str,
    model: QuantizableMobileNetV3,
    model_url: Optional[str],
    progress: bool,
):
    if model_url is None:
        raise ValueError("No checkpoint is available for {}".format(arch))
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    quantize: bool,
    **kwargs: Any
):
    model = QuantizableMobileNetV3(inverted_residual_setting, last_channel, block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        backend = 'qnnpack'

        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)

        if pretrained:
            _load_weights(arch, model, quant_model_urls.get(arch + '_' + backend, None), progress)

        torch.quantization.convert(model, inplace=True)
        model.eval()
    else:
        if pretrained:
            _load_weights(arch, model, model_urls.get(arch, None), progress)

    return model


def mobilenet_v3_large(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a MobileNetV3 Large architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
     pretrained (bool): If True, returns a model pre-trained on ImageNet.
     progress (bool): If True, displays a progress bar of the download to stderr
     quantize (bool): If True, returns a quantized model, else returns a float model
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, quantize, **kwargs)
