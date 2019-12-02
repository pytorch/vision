from torch import nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU, MobileNetV2, model_urls
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from .utils import _replace_relu, quantize_model


__all__ = ['QuantizableMobileNetV2', 'mobilenet_v2']

quant_model_urls = {
    'mobilenet_v2_qnnpack':
        'https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth'
}


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()


def mobilenet_v2(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
     pretrained (bool): If True, returns a model pre-trained on ImageNet.
     progress (bool): If True, displays a progress bar of the download to stderr
     quantize(bool): If True, returns a quantized model, else returns a float model
    """
    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'qnnpack'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls['mobilenet_v2_' + backend]
        else:
            model_url = model_urls['mobilenet_v2']

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)
    return model
