# TODO: Implement v1 and v2 versions of the mobile ViT model.

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv2 import InvertedResidual

from torchvision.ops.misc import MLP
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once

__all__ = ["MobileViT", "MobileViT_Weights"]

_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}




# The EncoderBlock and Encoder from vision_transformer.py
# TODO: Maybe refactor later...
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,  # This is the embedding dim (known as E or d), should be a multiple of num_heads...
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block (inspired from swin_transformer.py)
        # TODO: Rename the hidden_dim variable...
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, input: torch.Tensor):
        # B x N x D
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.mlp(x)
        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,  # This is the depth... Okay...
        num_heads: int,  # This is number of heads in the multi-attention layer... Okay ...
        hidden_dim: int,  # This is the embedding or d dimension, should be a multiple of num_heads...
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        # Multiple iteration over the num_layers/depth...
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

    def forward(self, x: torch.Tensor):
        tensors = []
        # Here we loop over the P pixels of the
        # tensor x of shape: B, P, N, d
        for p in range(x.shape[1]):
            tmp_tensor = self.layers(x[:, p, :, :])
            # Adding back the patch dimension before concatenating
            tmp_tensor = tmp_tensor.unsqueeze(1)
            tensors.append(tmp_tensor)
        return torch.cat(tensors, dim=1)


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        channel,
        kernel_size,
        patch_size: Tuple[int, int],
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.5,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False), nn.BatchNorm2d(channel), nn.SiLU()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(channel, dim, 1, 1, 0, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        num_heads = 4
        self.transformer = TransformerEncoder(depth, num_heads, dim, mlp_dim, dropout, attention_dropout)
        self.conv3 = nn.Sequential(nn.Conv2d(dim, channel, 1, 1, 0, bias=False), nn.BatchNorm2d(channel), nn.SiLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size, 1, 1, bias=False), nn.BatchNorm2d(channel), nn.SiLU()
        )

    @staticmethod
    def _unfold(x: Tensor, patch_size: Tuple[int, int], n_patches: Tuple[int, int]) -> Tensor:
        """
        Unfold a batch of B image tensors B x d x H X W into a batch of B  P x N x d tensors
        (N is the number of patches)
        These P x N x d tensors are then used by the transformer encoder where d is the hidden
        dimension/encoding, N is the sequence length and we loop over the pixels P.
        """
        h_patch, w_patch = patch_size
        n_h_patch, n_w_patch = n_patches
        # P is the number of pixels
        P = h_patch * w_patch
        B, d, _, _ = x.shape
        N = n_w_patch * n_h_patch

        # We reshape from B x d x H x W to (B * d * n_h_patch) x h_patch x n_w_patch x w_patch
        x = x.reshape(B * d * n_w_patch, h_patch, n_h_patch, w_patch)
        # Then we transpose  (B * d * n_h_patch) x h_patch x n_w_patch x w_patch into (B * d * n_h_patch) x  n_w_patch x h_patch x w_patch
        x = x.transpose(1, 2)
        # Next, we reshape (B * d * n_h_patch) x  n_w_patch x h_patch x w_patch into B x d x N x P
        x = x.reshape(B, d, N, P)
        # And we finish by transposing B x d x N x P into B x P x N x d
        x = x.transpose(1, 3)
        return x

    @staticmethod
    def _fold(x: Tensor, patch_size: Tuple[int, int], n_patches: Tuple[int, int]) -> Tensor:
        """
        Fold a batch of B  P x N x d tensors
        (N is the number of patches) into a batch of B d x H x W image tensors.
        This is the reverse operation of unfold.
        """
        h_patch, w_patch = patch_size
        n_h_patch, n_w_patch = n_patches
        B, _, _, d = x.shape
        x = x.transpose(1, 3)

        x = x.reshape(B * d * n_h_patch, n_w_patch, h_patch, w_patch)
        x = x.transpose(1, 2)
        x = x.reshape(B, d, n_h_patch * h_patch, n_w_patch * w_patch)
        return x

    def forward(self, x):
        # We compute how many patches along the width patch dimension, the height patch dimension,
        # and the total number of patches.
        # The number of patches N x the numbre of pixels P in a patch
        # is equal to the image area H x W.
        _, _, H, W = x.shape
        h_patch, w_patch = self.patch_size
        n_w_patch = W // w_patch
        n_h_patch = H // h_patch
        n_patches = (n_h_patch, n_w_patch)
        y = x.detach().clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self._unfold(x, patch_size=self.patch_size, n_patches=n_patches)
        # We get a tensor of shape: B x P x N x d after the previous steps
        x = self.transformer(x)
        # The transformer blocks keep the B x P x N x d shape
        x = self._fold(x, patch_size=self.patch_size, n_patches=n_patches)
        # We get back B x d x H x W tensors
        x = self.conv3(x)
        # Then we get the inital shape B x C x H X W
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    """
    Implements MobileViT from the `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_ paper.
    Args:
        num_classes (int): Number of classes for classification head. Default: 1000.
        d (List[int]): A list of the layers' dimensions.
        c (List[int]): A list of the layers' channels.
        expand_ratio (int): The expansion ratio of the InvertedResidual block. Default: 4.
    """

    def __init__(self, num_classes: int = 1000, d: List[int] = None, c: List[int] = None, expand_ratio: int = 4):
        super().__init__()
        _log_api_usage_once(self)
        if len(d) != 3:
            raise ValueError(f"d should be non-empty list, got {d}")
        if len(c) != 11:
            raise ValueError(f"c should be non-empty list, got {c}")
        self.num_classes = num_classes
        self.expand_ratio = expand_ratio
        # n x n convolution as an input layer
        # 3 is the number of RGB channels thus it is the
        # input dimension.
        self.conv_first = nn.Sequential(nn.Conv2d(3, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.SiLU())
        self.transformer_depths = [2, 4, 3]
        layers = [
            InvertedResidual(inp=c[0], oup=c[1], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(inp=c[1], oup=c[2], stride=2, expand_ratio=self.expand_ratio),
            # Twice the same block used here.
            InvertedResidual(inp=c[2], oup=c[3], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(inp=c[2], oup=c[3], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(inp=c[3], oup=c[4], stride=2, expand_ratio=self.expand_ratio),
            MobileViTBlock(
                dim=d[0],
                channel=c[5],
                depth=self.transformer_depths[0],
                kernel_size=3,
                patch_size=(2, 2),
                mlp_dim=d[0] * 2,
            ),
            InvertedResidual(inp=c[5], oup=c[6], stride=2, expand_ratio=self.expand_ratio),
            MobileViTBlock(
                dim=d[1],
                channel=c[7],
                depth=self.transformer_depths[1],
                kernel_size=3,
                patch_size=(2, 2),
                mlp_dim=d[1] * 4,
            ),
            InvertedResidual(inp=c[7], oup=c[8], stride=2, expand_ratio=self.expand_ratio),
            MobileViTBlock(
                dim=d[2],
                channel=c[9],
                depth=self.transformer_depths[2],
                kernel_size=3,
                patch_size=(2, 2),
                mlp_dim=d[2] * 4,
            ),
        ]
        self.features = nn.Sequential(*layers)
        # height // 32 gives 8 for height 256...
        self.avgpool = nn.AvgPool2d(8, 1)
        # 1 x 1 convolution as an output layer (before fc)
        self.conv_last = nn.Sequential(nn.Conv2d(c[9], c[10], 1, 1, 0, bias=False), nn.BatchNorm2d(c[10]), nn.SiLU())
        self.classifier = nn.Sequential( 
            nn.Flatten(1), nn.Linear(c[10], self.num_classes) 
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv_last(x)
        x = self.classifier(x)
        return x


def _mobile_vit(
    num_classes: int,
    d: List[int],
    c: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    expand_ratio: int = 4,
    **kwargs: Any,
) -> MobileViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileViT(
        num_classes=num_classes,
        c=c,
        d=d,
        expand_ratio=expand_ratio,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


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


@register_model()
def mobile_vit_s(*, weights: Optional[MobileViT_Weights] = None, progress: bool = True, **kwargs: Any):
    """
    Constructs a mobile_vit_s architecture from
    `"MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transfomer" <https://arxiv.org/abs/2110.02178>`_.

    Args:
        weights (:class:`~torchvision.models.MobileViT_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileViT_Weights` below for
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
    s_c = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    s_d = [144, 192, 240]
    weights = MobileViT_Weights.verify(weights)
    return _mobile_vit(c=s_c, d=s_d, weights=weights, progress=progress, **kwargs)


@register_model()
def mobile_vit_xs(*, weights: Optional[MobileViT_Weights] = None, progress: bool = True, **kwargs: Any):
    # TODO: Add the documentation
    xs_c = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    xs_d = [96, 120, 144]
    weights = MobileViT_XS_Weights.verify(weights)
    return _mobile_vit(c=xs_c, d=xs_d, weights=weights, progress=progress, **kwargs)


@register_model()
def mobile_vit_xxs(*, weights: Optional[MobileViT_Weights] = None, progress: bool = True, **kwargs: Any):
    # TODO: Add the documentation
    xxs_c = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    xxs_d = [64, 80, 96]
    weights = MobileViT_XXS_Weights.verify(weights)
    return _mobile_vit(c=xxs_c, d=xxs_d, weights=weights, progress=progress, expand_ratio=2, **kwargs)



if __name__ == "__main__":
    block = MobileViTBlock(dim=8 * 10, depth=1, channel=3, kernel_size=3, patch_size=(2, 2), mlp_dim=2, dropout=0.5)
    # B x C x H x W
    x = torch.rand(10, 3, 10, 10)
    assert block(x).shape == (10, 3, 10, 10)

    # Batch of 10 RGB (256 x 256) random images
    img = torch.randn(10, 3, 256, 256)
    model = mobile_vit_s(num_classes=1000)
    assert model(img).shape == (10, 1000)
