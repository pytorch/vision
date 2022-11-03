# TODO: Implement v1 and v2 versions of the mobile ViT model.

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.utils import _log_api_usage_once

from torchvision.ops.misc import MLP
from torchvision.transforms._presets import ImageClassification

__all__ = ["MobileViT", "MobileViT_Weights", "MobileViT_V2_Weights"]

_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}

# For V1, we have 3 sets of weights xx_small (1.3M parameters), x_small (2.3M parameters), and small (5.6M parameters)
# For V2, we have one set of weights.
# Paper link: v1 https://arxiv.org/abs/2110.02178.
# Paper link: v2 https://arxiv.org/pdf/2206.02680.pdf.
# v2 (what the difference with the V1 paper?)
# Things to be done: write the V1, MobileViTblock, MobileViTV2block, weights (for V1 and V2), documentation...
# TODO: What about multi-scale sampler? Check later...


class MobileViT_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        # TODO: Update the URL once the model has been trained...
        url="https://download.pytorch.org/models/mobilevit.pth",
        transforms=partial(ImageClassification, crop_size=256),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilevit",
            "_metrics": {
                # TODO: Update with the correct values. For now, these are the expected ones from the paper.
                "ImageNet-1K": {
                    "acc@1": 78.4,
                    "acc@5": 94.1,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class MobileViT_XS_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        # TODO: Update the URL once the model has been trained...
        url="https://download.pytorch.org/models/mobilevit_xs.pth",
        transforms=partial(ImageClassification, crop_size=256),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilevit",
            "_metrics": {
                # TODO: Update with the correct values. For now, these are the expected ones from the paper.
                "ImageNet-1K": {
                    "acc@1": 74.8,
                    "acc@5": 92.3,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class MobileViT_XXS_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        # TODO: Update the URL once the model has been trained...
        url="https://download.pytorch.org/models/mobilevit_xxs.pth",
        transforms=partial(ImageClassification, crop_size=256),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilevit",
            "_metrics": {
                # TODO: Update with the correct values. For now, these are the expected ones from the paper.
                "ImageNet-1K": {
                    "acc@1": 69.0,
                    "acc@5": 88.9,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


# TODO: Take inspiration from the V1 weights... In progress...
class MobileViT_V2_Weights(WeightsEnum):
    pass


# The EncoderBlock and Encoder from vision_transformer.py
# TODO: Maybe refactor later...
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block (inspired from swin_transformer.py)
        self.mlp = MLP(mlp_dim, [hidden_dim, mlp_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.mlp(y)
        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        # Multiple
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = TransformerEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


# TODO: We will need a mobilenet block as well.
# TODO: We need to use a Transformer. In progress... Using the one from TorchVision...
# TODO: We need a LayerNorm as well...In progress...


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_dimensions, mlp_dim, dropout=0.0):
        super().__init__()
        self.patch_height, self.patch_width = patch_dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, bias=False), nn.BatchNorm2d(channel), nn.SiLU()
        )
        # Point-wise convolution (1 x 1)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, dim, 1, 1, 0, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        # TODO: Setup the inputs...
        self.transformer = TransformerEncoder(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = nn.Sequential(nn.Conv2d(dim, channel, 1, 1, 0, bias=False), nn.BatchNorm2d(channel), nn.SiLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size, 1, bias=False), nn.BatchNorm2d(channel), nn.SiLU()
        )

    def forward(self, x):
        y = x.copy()
        x = self.conv1(x)
        x = self.conv2(x)
        # batch, channels, height, width.
        _, _, h, w = x.shape
        # This is the unfloding (from spatial features to patches) and folding (from patches back to features) parts.
        # TODO: What are the values of self.ph and self.pw.
        # TODO: Change with a PyTorch operation... In progress...
        print(x.shape)
        """
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        # The reverse operation...
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        """
        return x


# Separable self-attention
# TODO: Is this necessary? Check... Maybe
class MobileViTV2Block(MobileViTBlock):
    def forward(self, x: Tensor):
        return x


class MobileViT(nn.Module):
    """
    Implements MobileViT from the `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_ paper.
    Args:
        TODO: Arguments to be updated... in progress
        num_classes (int): Number of classes for classification head. Default: 1000.
        layers_conf (dict): The layers configuration.
    """

    def __init__(
        self,
        # Trained on ImageNet1K by default.
        num_classes: int = 1000,
        layers_conf: dict = None,
        # TODO: Should this be optional? Yes probably...
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        # TODO: Add blocks... In progress...
        self.num_classes = num_classes

        if block is None:
            block = MobileViTBlock
        # Build the model one layer at a time.
        layers: List[nn.Module] = []
        self.features = nn.Sequential(*layers)

    # TODO: This is the core thing to implement...
    def forward(self, x):
        x = self.features(x)
        return x


def _mobile_vit(
    # TODO: Update the parameters...
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileViT(
        # TODO: Update these...Will pass different configurations depending on the size of the mdoel...
        # In progress...
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@register_model()
def mobile_vit_s(*, weights: Optional[MobileViT_Weights] = None, progress: bool = True, **kwargs: Any):
    """
    Constructs a mobile_vit_s architecture from
    `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_.

    Args:
        weights (:class:`~torchvision.models.MobileViT_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobile_vit.MobileVit``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilevit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileViT_Weights
        :members:
    """
    weights = MobileViT_Weights.verify(weights)
    return _mobile_vit(weights=weights)


@register_model()
def mobile_vit_xs():
    weights = MobileViT_XS_Weights.verify(weights)
    return _mobile_vit(weights=weights)


@register_model()
def mobile_vit_xxs():
    weights = MobileViT_XXS_Weights.verify(weights)
    return _mobile_vit(weights=weights)


@register_model()
def mobile_vit_v2():
    weights = MobileViT_V2_Weights.verify(weights)
    return _mobile_vit(weights=weights)


if __name__ == "__main__":
    print(MobileViTBlock(1, 3, 1, 1, 0.5))
