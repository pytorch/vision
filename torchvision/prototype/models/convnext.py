from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...ops.misc import ConvNormActivation
from ...ops.stochastic_depth import StochasticDepth
from ...utils import _log_api_usage_once
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["ConvNeXt", "ConvNeXt_Tiny_Weights", "convnext_tiny"]


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.channels_last = kwargs.pop("channels_last", False)
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        #  TODO: Benchmark this against the approach described at https://github.com/pytorch/vision/pull/5197#discussion_r786251298
        if not self.channels_last:
            x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if not self.channels_last:
            x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self, dim, layer_scale: float, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module]
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormActivation(
                dim,
                dim,
                kernel_size=7,
                groups=dim,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            ),
            ConvNormActivation(dim, 4 * dim, kernel_size=1, norm_layer=None, activation_layer=nn.GELU, inplace=None),
            ConvNormActivation(
                4 * dim,
                dim,
                kernel_size=1,
                norm_layer=None,
                activation_layer=None,
            ),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
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
        out_channels: Optional[int],
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
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-47b116bd.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=236),
        meta={
            "task": "image_classification",
            "architecture": "ConvNeXt",
            "publication_year": 2022,
            "num_params": 28589128,
            "size": (224, 224),
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "acc@1": 82.520,
            "acc@5": 96.146,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (ConvNeXt_Tiny_Weights, optional): The pre-trained weights of the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
