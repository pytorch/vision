import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception as inception_module
from torchvision.models.inception import InceptionOutputs
from torch.jit.annotations import Optional
from torchvision.models.utils import load_state_dict_from_url
from .utils import _replace_relu, quantize_model


__all__ = [
    "QuantizableInception3",
    "inception_v3",
]


quant_model_urls = {
    # fp32 weights ported from TensorFlow, quantized in PyTorch
    "inception_v3_google_fbgemm":
        "https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth"
}


def inception_v3(pretrained=False, progress=True, quantize=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Note that quantize = True returns a quantized model with 8 bit
    weights. Quantized models only support inference and run on CPUs.
    GPU inference is not yet supported

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" in kwargs:
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
        else:
            original_aux_logits = False

    model = QuantizableInception3(**kwargs)
    _replace_relu(model)

    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            if not original_aux_logits:
                model.aux_logits = False
                del model.AuxLogits
            model_url = quant_model_urls['inception_v3_google' + '_' + backend]
        else:
            model_url = inception_module.model_urls['inception_v3_google']

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)

        if not quantize:
            if not original_aux_logits:
                model.aux_logits = False
                del model.AuxLogits
    return model


class QuantizableBasicConv2d(inception_module.BasicConv2d):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicConv2d, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableInceptionA(inception_module.InceptionA):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionA, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionB(inception_module.InceptionB):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionB, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionC(inception_module.InceptionC):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionC, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionD(inception_module.InceptionD):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionD, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionE(inception_module.InceptionE):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionE, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)
        self.myop1 = nn.quantized.FloatFunctional()
        self.myop2 = nn.quantized.FloatFunctional()
        self.myop3 = nn.quantized.FloatFunctional()

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = self.myop1.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = self.myop2.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop3.cat(outputs, 1)


class QuantizableInceptionAux(inception_module.InceptionAux):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionAux, self).__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)


class QuantizableInception3(inception_module.Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(QuantizableInception3, self).__init__(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input,
            inception_blocks=[
                QuantizableBasicConv2d,
                QuantizableInceptionA,
                QuantizableInceptionB,
                QuantizableInceptionC,
                QuantizableInceptionD,
                QuantizableInceptionE,
                QuantizableInceptionAux
            ]
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableInception3 always returns QuantizableInception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in inception model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                m.fuse_model()
