import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence

from ..ops.misc import ConvNormActivation
from ..ops.stochastic_depth import StochasticDepth
from ..utils import _log_api_usage_once


class CNBlock(nn.Module):
    def __init__(self, dim,
                 stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module],
                 layer_scale: Optional[float] = 1e-6):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormActivation(
                dim,
                dim,
                kernel_size=7,
                groups=dim,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,  # TODO: check
            ),
            ConvNormActivation(
                dim,
                4 * dim,
                kernel_size=1,
                norm_layer=None,
                activation_layer=nn.GELU,
            ),
            ConvNormActivation(
                4 * dim,
                dim,
                kernel_size=1,
                norm_layer=None,
                activation_layer=None,
            )
        )
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)



class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        layers: List[nn.Module] = [

        ]
