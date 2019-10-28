import warnings
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import googlenet as googlenet_module
from torchvision.models.googlenet import GoogLeNetOutputs
import sys
from torch import Tensor
from torch.jit.annotations import Optional
from .utils import _replace_relu, quantize_model


__all__ = ['QuantizableGoogLeNet', 'googlenet']

quant_model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': '',
}


def googlenet(pretrained=False, progress=True, quantize=False, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False

    model = QuantizableGoogLeNet(**kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls['googlenet']
        else:
            model_url = googlenet_module.model_urls['googlenet']

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)

        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
    return model


class QuantizableBasicConv2d(googlenet_module.BasicConv2d):

    def __init__(self, *args, **kwargs):
        super(QuantizableBasicConv2d, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class QuantizableInception(googlenet_module.Inception):

    def __init__(self, *args, **kwargs):
        super(QuantizableInception, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.cat.cat(outputs, 1)


class QuantizableGoogLeNet(googlenet_module.GoogLeNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableGoogLeNet, self).__init__(
            basic_conv2d=QuantizableBasicConv2d,
            inception=QuantizableInception,
            *args,
            **kwargs
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux1, aux2 = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableGoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
