# Implement ViT from:
# https://arxiv.org/abs/2010.11929

# References:
# https://github.com/google-research/vision_transformer
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/vision_transformer.py

import math
from collections import OrderedDict
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "VisionTransformer",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
]


LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout_rate: float, attention_dropout_rate: float
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # MLP block
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)

    def forward(self, input: Tensor):
        # assert input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}"
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()
        # Note that batch_size is on the second dim because
        # we have batch_first=False in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout_rate)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout_rate,
                attention_dropout_rate,
            )
        self.layers = nn.Sequential(layers)
        self.ln = LayerNorm(hidden_dim)

    def forward(self, input: Tensor):
        # assert input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}"
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        classifier: str = "token",
        num_classes: int = 1000,
    ):
        super().__init__()
        # assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        # assert classifier in ["token", "gap"], "Unexpected classifier mode!"
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.classifier = classifier
        self.num_classes = num_classes

        input_channels = 3

        # The conv_proj is a more efficient version of reshaping, permuting
        # and projecting the input
        self.conv_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # Add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
        )
        self.seq_length = seq_length

        self.head = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.head.weight)

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        p = self.patch_size
        # assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # The self attention layer expects inputs in the format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # Expand the class token to the full batch.
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x = self.encoder(x)

        if self.classifier == "token":
            # Classifier as used by standard language architectures
            x = x[0, :, :]
        elif self.classifier == "gap":
            # Classifier as used by standard vision architectures
            x = x.mean(dim=0)
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        x = self.head(x)

        return x


def _vision_transformer(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> VisionTransformer:
    if kwargs.get("image_size", None) is None:
        model = VisionTransformer(image_size=224, **kwargs)
    else:
        model = VisionTransformer(**kwargs)
    # TODO: Adding pre-trained models
    return model


def vit_b_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT_b_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        version="b_16",
        pretrained=pretrained,
        progress=progress,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


def vit_b_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT_b_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        version="b_32",
        pretrained=pretrained,
        progress=progress,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


def vit_l_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT_l_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        version="l_16",
        pretrained=pretrained,
        progress=progress,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


def vit_l_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT_l_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        version="l_32",
        pretrained=pretrained,
        progress=progress,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )
