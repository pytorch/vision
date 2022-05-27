import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
import torch.fx
import torch.nn as nn

from ...ops import StochasticDepth, MLP
from .._api import WeightsEnum
from .._utils import _ovewrite_named_param


__all__ = ["mvitv2_t", "mvitv2_s", "mvitv2_b", "MViTV2_T_Weights", "MViTV2_S_Weights", "MViTV2_B_Weights"]


# TODO: add docs
# TODO: add weights
# TODO: test on references


def _prod(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product


def _unsqueeze(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == 3:
        x = x.unsqueeze(1)
    elif tensor_dim != 4:
        raise NotImplementedError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


def _squeeze(x: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == 3:
        x = x.squeeze(1)
    return x


torch.fx.wrap("_unsqueeze")
torch.fx.wrap("_squeeze")


class Pool(nn.Module):
    def __init__(
        self,
        pool: nn.Module,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module] = None,
        norm_before_pool: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        layers = []
        if norm is not None:
            layers.append(norm)
        if activation is not None:
            layers.append(activation)
        self.norm_act = nn.Sequential(*layers) if layers else None
        self.norm_before_pool = norm_before_pool

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x, tensor_dim = _unsqueeze(x)

        # Separate the class token and reshape the input
        class_token, x = torch.tensor_split(x, indices=(1,), dim=2)
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw)

        # normalizing prior pooling is useful when we use BN which can be absorbed to speed up inference
        if self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        # apply the pool on the input and add back the token
        x = self.pool(x)
        T, H, W = x.shape[2:]
        x = x.reshape(B, N, C, -1).transpose(2, 3)
        x = torch.cat((class_token, x), dim=2)

        if not self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        x = _squeeze(x, tensor_dim)
        return x, (T, H, W)


class MultiscaleAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kernel_q: Tuple[int, int, int] = (1, 1, 1),
        kernel_kv: Tuple[int, int, int] = (1, 1, 1),
        stride_q: Tuple[int, int, int] = (1, 1, 1),
        stride_kv: Tuple[int, int, int] = (1, 1, 1),
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        layers: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)

        self.pool_q: Optional[nn.Module] = None
        if _prod(kernel_q) > 1 or _prod(stride_q) > 1:
            padding_q = cast(Tuple[int, int, int], tuple(int(q // 2) for q in kernel_q))
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _prod(kernel_kv) > 1 or _prod(stride_kv) > 1:
            padding_kv = cast(Tuple[int, int, int], tuple(int(kv // 2) for kv in kernel_kv))
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)

        if self.pool_k is not None:
            k = self.pool_k(k, thw)[0]
        if self.pool_v is not None:
            v = self.pool_v(v, thw)[0]
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw)

        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v).add_(q)
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.project(x)

        return x, thw


class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_heads: int,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        kernel_q: Tuple[int, int, int] = (1, 1, 1),
        kernel_kv: Tuple[int, int, int] = (1, 1, 1),
        stride_q: Tuple[int, int, int] = (1, 1, 1),
        stride_kv: Tuple[int, int, int] = (1, 1, 1),
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.pool_skip: Optional[nn.Module] = None
        if _prod(stride_q) > 1:
            kernel_skip = cast(Tuple[int, int, int], tuple(s + 1 if s > 1 else s for s in stride_q))
            padding_skip = cast(Tuple[int, int, int], tuple(int(k // 2) for k in kernel_skip))
            self.pool_skip = Pool(nn.MaxPool3d(kernel_skip, stride=stride_q, padding=padding_skip), None)

        self.norm1 = norm_layer(input_channels)
        self.norm2 = norm_layer(input_channels)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)

        self.attn = MultiscaleAttention(
            input_channels,
            num_heads,
            dropout=dropout,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(
            input_channels,
            [4 * input_channels, output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.project: Optional[nn.Module] = None
        if input_channels != output_channels:
            self.project = nn.Linear(input_channels, output_channels)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x_skip = x if self.pool_skip is None else self.pool_skip(x, thw)[0]

        x = self.norm1(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm1(x)
        x, thw = self.attn(x, thw)
        x = x_skip + self.stochastic_depth(x)

        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm2(x)
        x_proj = x if self.project is None else self.project(x_norm)

        return x_proj + self.stochastic_depth(self.mlp(x_norm)), thw


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        self.class_token = nn.Parameter(torch.zeros(embed_size))
        self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_size[0] * self.spatial_size[1], embed_size))
        self.temporal_pos = nn.Parameter(torch.zeros(self.temporal_size, embed_size))
        self.class_pos = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw_size, embed_size = self.spatial_pos.shape
        pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
        pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.temporal_size, -1, -1).reshape(-1, embed_size))
        pos_embedding = torch.cat((self.class_pos.unsqueeze(0), pos_embedding), dim=0).unsqueeze(0)
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        return torch.cat((class_token, x), dim=1) + pos_embedding


class MultiscaleVisionTransformer(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        temporal_size: int,
        embed_channels: List[int],
        blocks: List[int],
        heads: List[int],
        pool_kv_stride: Tuple[int, int, int] = (1, 8, 8),
        pool_q_stride: Tuple[int, int, int] = (1, 2, 2),
        pool_kvq_kernel: Tuple[int, int, int] = (3, 3, 3),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=3,
            out_channels=embed_channels[0],
            kernel_size=(3, 7, 7),
            stride=(2, 4, 4),
            padding=(1, 3, 3),
        )

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=embed_channels[0],
            spatial_size=(spatial_size[0] // self.conv_proj.stride[1], spatial_size[1] // self.conv_proj.stride[2]),
            temporal_size=temporal_size // self.conv_proj.stride[0],
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        stage_block_id = 0
        pool_countdown = blocks[0]
        input_channels = output_channels = embed_channels[0]
        stride_kv = pool_kv_stride
        total_stage_blocks = sum(blocks)
        for i, num_subblocks in enumerate(blocks):
            for j in range(num_subblocks):
                next_block_index = i + 1 if j + 1 == num_subblocks and i + 1 < len(embed_channels) else i
                output_channels = embed_channels[next_block_index]

                stride_q = (1, 1, 1)
                if pool_countdown == 0:
                    stride_q = pool_q_stride
                    pool_countdown = blocks[next_block_index]

                stride_kv = cast(Tuple[int, int, int], tuple(max(s // stride_q[d], 1) for d, s in enumerate(stride_kv)))

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

                self.blocks.append(
                    block(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        num_heads=heads[i],
                        dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        kernel_q=pool_kvq_kernel,
                        kernel_kv=pool_kvq_kernel,
                        stride_q=stride_q,
                        stride_kv=stride_kv,
                        norm_layer=norm_layer,
                    )
                )
                input_channels = output_channels
                stage_block_id += 1
                pool_countdown -= 1
        self.norm = norm_layer(output_channels)

        # Classifier module
        layers: List[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(nn.Linear(output_channels, num_classes))
        self.head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patchify and reshape: (B, C, T, H, W) -> (B, patch_embed_dim, T', H', W') -> (B, THW', patch_embed_dim)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return x


def _mvitv2(
    temporal_size: int,
    embed_channels: List[int],
    blocks: List[int],
    heads: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MultiscaleVisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"][0])
        # TODO: add min_temporal_size in the meta-data?
    spatial_size = kwargs.pop("spatial_size", (224, 224))

    model = MultiscaleVisionTransformer(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        embed_channels=embed_channels,
        blocks=blocks,
        heads=heads,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class MViTV2_T_Weights(WeightsEnum):
    pass


class MViTV2_S_Weights(WeightsEnum):
    pass


class MViTV2_B_Weights(WeightsEnum):
    pass


def mvitv2_t(
    *, weights: Optional[MViTV2_T_Weights] = None, progress: bool = True, **kwargs: Any
) -> MultiscaleVisionTransformer:
    weights = MViTV2_T_Weights.verify(weights)

    return _mvitv2(
        spatial_size=(224, 224),
        temporal_size=16,
        embed_channels=[96, 192, 384, 768],
        blocks=[1, 2, 5, 2],
        heads=[1, 2, 4, 8],
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.1),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def mvitv2_s(
    *, weights: Optional[MViTV2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> MultiscaleVisionTransformer:
    weights = MViTV2_S_Weights.verify(weights)

    return _mvitv2(
        spatial_size=(224, 224),
        temporal_size=16,
        embed_channels=[96, 192, 384, 768],
        blocks=[1, 2, 11, 2],
        heads=[1, 2, 4, 8],
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.1),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def mvitv2_b(
    *, weights: Optional[MViTV2_B_Weights] = None, progress: bool = True, **kwargs: Any
) -> MultiscaleVisionTransformer:
    weights = MViTV2_B_Weights.verify(weights)

    return _mvitv2(
        spatial_size=(224, 224),
        temporal_size=32,
        embed_channels=[96, 192, 384, 768],
        blocks=[2, 3, 16, 3],
        heads=[1, 2, 4, 8],
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.3),
        weights=weights,
        progress=progress,
        **kwargs,
    )
