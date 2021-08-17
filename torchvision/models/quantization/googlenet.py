import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any
from torch import Tensor

from ..._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.googlenet import (
    GoogLeNetOutputs, BasicConv2d, Inception, InceptionAux, GoogLeNet, model_urls)

from .utils import _replace_relu, quantize_model


__all__ = ['QuantizableGoogLeNet', 'googlenet']

quant_model_urls = {
    # fp32 GoogLeNet ported from TensorFlow, with weights quantized in PyTorch
    'googlenet_fbgemm': 'https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth',
}


def googlenet(
    pretrained: bool = False,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> "QuantizableGoogLeNet":

    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
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
            model_url = quant_model_urls['googlenet' + '_' + backend]
        else:
            model_url = model_urls['googlenet']

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)

        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
    return model


class QuantizableBasicConv2d(BasicConv2d):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuantizableBasicConv2d, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self) -> None:
        torch.quantization.fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableInception(Inception):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuantizableInception, self).__init__(  # type: ignore[misc]
            conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat.cat(outputs, 1)


class QuantizableInceptionAux(InceptionAux):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuantizableInceptionAux, self).__init__(  # type: ignore[misc]
            conv_block=QuantizableBasicConv2d,
            *args,
            **kwargs
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.relu(self.fc1(x))
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class QuantizableGoogLeNet(GoogLeNet):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuantizableGoogLeNet, self).__init__(  # type: ignore[misc]
            blocks=[QuantizableBasicConv2d, QuantizableInception, QuantizableInceptionAux],
            *args,
            **kwargs
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
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

    def fuse_model(self) -> None:
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                m.fuse_model()
