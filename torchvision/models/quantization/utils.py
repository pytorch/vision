from enum import Enum
from typing import Any, List, Optional, Union

import torch
import torch.ao.quantization.quantize_fx
from torch import nn


class QuantizationWorkflowType(Enum):
    EAGER_MODE = "eager_mode"
    FX_GRAPH_MODE = "fx_graph_mode"


def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def quantize_model(model: nn.Module, backend: str, quantization_workflow_type: QuantizationWorkflowType) -> nn.Module:
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    if quantization_workflow_type == QuantizationWorkflowType.EAGER_MODE:
        # Make sure that weight qconfig matches that of the serialized models
        if backend == "fbgemm":
            model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_observer,
                weight=torch.ao.quantization.default_per_channel_weight_observer,
            )
        elif backend == "qnnpack":
            model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
                activation=torch.ao.quantization.default_observer, weight=torch.ao.quantization.default_weight_observer
            )

        # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
        model.fuse_model()  # type: ignore[operator]
        torch.ao.quantization.prepare(model, inplace=True)
        model(_dummy_input_data)
        torch.ao.quantization.convert(model, inplace=True)
    elif quantization_workflow_type == QuantizationWorkflowType.FX_GRAPH_MODE:
        qconfig_dict = torch.ao.quantization.get_default_qconfig_dict(backend)
        model = torch.ao.quantization.quantize_fx.prepare_fx(model, qconfig_dict)
        model(_dummy_input_data)
        model = torch.ao.quantization.quantize_fx.convert_fx(model)
    else:
        raise ValueError("Unknown quantization workflow type '%s'" % quantization_workflow_type)
    return model


def _fuse_modules(
    model: nn.Module, modules_to_fuse: Union[List[str], List[List[str]]], is_qat: Optional[bool], **kwargs: Any
):
    if is_qat is None:
        is_qat = model.training
    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    return method(model, modules_to_fuse, **kwargs)
