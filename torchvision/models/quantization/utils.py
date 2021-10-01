import torch
from torch import nn

from typing import Tuple
TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])

if TORCH_VERSION >= (1, 10):
    import torch.ao.quantization as tq
else:
    import torch.quantization as tq


def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def quantize_model(model: nn.Module, backend: str) -> None:
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = tq.QConfig(  # type: ignore[assignment]
            activation=tq.default_observer,
            weight=tq.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = tq.QConfig(  # type: ignore[assignment]
            activation=tq.default_observer,
            weight=tq.default_weight_observer)

    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    model.fuse_model()  # type: ignore[operator]
    tq.prepare(model, inplace=True)
    model(_dummy_input_data)
    tq.convert(model, inplace=True)

    return
