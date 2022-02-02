import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import inception as inception_module
from torchvision.models.inception import InceptionOutputs

from ..._internally_replaced_utils import load_state_dict_from_url
from .utils import _fuse_modules, _replace_relu, quantize_model


__all__ = [
    "QuantizableInception3",
    "inception_v3",
]


quant_model_urls = {
    # fp32 weights ported from TensorFlow, quantized in PyTorch
    "inception_v3_google_fbgemm": "https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth"
}


class QuantizableBasicConv2d(inception_module.BasicConv2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self, ["conv", "bn", "relu"], is_qat, inplace=True)


class QuantizableInceptionA(inception_module.InceptionA):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionB(inception_module.InceptionB):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionC(inception_module.InceptionC):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionD(inception_module.InceptionD):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionE(inception_module.InceptionE):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]
        self.myop1 = nn.quantized.FloatFunctional()
        self.myop2 = nn.quantized.FloatFunctional()
        self.myop3 = nn.quantized.FloatFunctional()

    def _forward(self, x: Tensor) -> List[Tensor]:
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

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.myop3.cat(outputs, 1)


class QuantizableInceptionAux(inception_module.InceptionAux):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(conv_block=QuantizableBasicConv2d, *args, **kwargs)  # type: ignore[misc]


class QuantizableInception3(inception_module.Inception3):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
    ) -> None:
        super().__init__(
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
                QuantizableInceptionAux,
            ],
        )
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> InceptionOutputs:
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

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        r"""Fuse conv/bn/relu modules in inception model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) is QuantizableBasicConv2d:
                m.fuse_model(is_qat)


def inception_v3(
    pretrained: bool = False,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableInception3:
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
        quantize (bool): If True, return a quantized version of the model
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: True if ``pretrained=True``, else False.
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
        backend = "fbgemm"
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            if not original_aux_logits:
                model.aux_logits = False
                model.AuxLogits = None
            model_url = quant_model_urls["inception_v3_google_" + backend]
        else:
            model_url = inception_module.model_urls["inception_v3_google"]

        state_dict = load_state_dict_from_url(model_url, progress=progress)

        model.load_state_dict(state_dict)

        if not quantize:
            if not original_aux_logits:
                model.aux_logits = False
                model.AuxLogits = None
    return model
