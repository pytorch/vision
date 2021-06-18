import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision.models.shufflenetv2
import sys
from .utils import _replace_relu, quantize_model

shufflenetv2 = sys.modules['torchvision.models.shufflenetv2']

__all__ = [
    'QuantizableShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

quant_model_urls = {
    'shufflenetv2_x0.5_fbgemm': None,
    'shufflenetv2_x1.0_fbgemm':
        'https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth',
    'shufflenetv2_x1.5_fbgemm': None,
    'shufflenetv2_x2.0_fbgemm': None,
}


class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.cat.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = self.cat.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = shufflenetv2.channel_shuffle(out, 2)

        return out


class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):
    def __init__(self, *args, **kwargs):
        super(QuantizableShuffleNetV2, self).__init__(*args, inverted_residual=QuantizableInvertedResidual, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in shufflenetv2 model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for name, m in self._modules.items():
            if name in ["conv1", "conv5"]:
                torch.quantization.fuse_modules(m, [["0", "1", "2"]], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableInvertedResidual:
                if len(m.branch1._modules.items()) > 0:
                    torch.quantization.fuse_modules(
                        m.branch1, [["0", "1"], ["2", "3", "4"]], inplace=True
                    )
                torch.quantization.fuse_modules(
                    m.branch2,
                    [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                    inplace=True,
                )


def _shufflenetv2(arch, pretrained, progress, quantize, *args, **kwargs):
    model = QuantizableShuffleNetV2(*args, **kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls[arch + '_' + backend]
        else:
            model_url = shufflenetv2.model_urls[arch]

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)
    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress, quantize,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress, quantize,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress, quantize,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, quantize=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress, quantize,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
