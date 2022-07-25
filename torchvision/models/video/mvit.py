import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx
import torch.nn as nn

from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param


__all__ = [
    "MViT",
    "MViT_V1_B_Weights",
    "mvit_v1_b",
]


# TODO: Consider handle 2d input if Temporal is 1


@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]


def _prod(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product


def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


def _squeeze(x: torch.Tensor, target_dim: int, expand_dim: int, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == target_dim - 1:
        x = x.squeeze(expand_dim)
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
        x, tensor_dim = _unsqueeze(x, 4, 1)

        # Separate the class token and reshape the input
        class_token, x = torch.tensor_split(x, indices=(1,), dim=2)
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw).contiguous()

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

        x = _squeeze(x, 4, 1, tensor_dim)
        return x, (T, H, W)


class MultiscaleAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_q: List[int],
        kernel_kv: List[int],
        stride_q: List[int],
        stride_kv: List[int],
        residual_pool: bool,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.residual_pool = residual_pool

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        layers: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)

        self.pool_q: Optional[nn.Module] = None
        if _prod(kernel_q) > 1 or _prod(stride_q) > 1:
            padding_q = [int(q // 2) for q in kernel_q]
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q,  # type: ignore[arg-type]
                    stride=stride_q,  # type: ignore[arg-type]
                    padding=padding_q,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _prod(kernel_kv) > 1 or _prod(stride_kv) > 1:
            padding_kv = [int(kv // 2) for kv in kernel_kv]
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
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

        x = torch.matmul(attn, v)
        if self.residual_pool:
            x.add_(q)
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.project(x)

        return x, thw


class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        cnf: MSBlockConfig,
        residual_pool: bool,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.pool_skip: Optional[nn.Module] = None
        if _prod(cnf.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(
                nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None  # type: ignore[arg-type]
            )

        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(cnf.input_channels)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)

        self.attn = MultiscaleAttention(
            cnf.input_channels,
            cnf.num_heads,
            kernel_q=cnf.kernel_q,
            kernel_kv=cnf.kernel_kv,
            stride_q=cnf.stride_q,
            stride_kv=cnf.stride_kv,
            residual_pool=residual_pool,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(
            cnf.input_channels,
            [4 * cnf.input_channels, cnf.output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.project: Optional[nn.Module] = None
        if cnf.input_channels != cnf.output_channels:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)

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
        return torch.cat((class_token, x), dim=1).add_(pos_embedding)


class MViT(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        temporal_size: int,
        block_setting: Sequence[MSBlockConfig],
        residual_pool: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
        """
        super().__init__()
        # This implementation employs a different parameterization scheme than the one used at PyTorch Video:
        # https://github.com/facebookresearch/pytorchvideo/blob/718d0a4/pytorchvideo/models/vision_transformers.py
        # We remove any experimental configuration that didn't make it to the final variants of the models. To represent
        # the configuration of the architecture we use the simplified form suggested at Table 1 of the paper.
        _log_api_usage_once(self)
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=3,
            out_channels=block_setting[0].input_channels,
            kernel_size=(3, 7, 7),
            stride=(2, 4, 4),
            padding=(1, 3, 3),
        )

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(spatial_size[0] // self.conv_proj.stride[1], spatial_size[1] // self.conv_proj.stride[2]),
            temporal_size=temporal_size // self.conv_proj.stride[0],
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                block(
                    cnf=cnf,
                    residual_pool=residual_pool,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )
        self.norm = norm_layer(block_setting[-1].output_channels)

        # Classifier module
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes),
        )

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
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
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


def _mvit(
    block_setting: List[MSBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"])
        _ovewrite_named_param(kwargs, "temporal_size", weights.meta["min_temporal_size"])
    spatial_size = kwargs.pop("spatial_size", (224, 224))
    temporal_size = kwargs.pop("temporal_size", 16)

    model = MViT(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=kwargs.pop("residual_pool", False),
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class MViT_V1_B_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url="https://download.pytorch.org/models/mvit_v1_b-dbeb1030.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
        ),
        meta={
            "min_size": (224, 224),
            "min_temporal_size": 16,
            "categories": _KINETICS400_CATEGORIES,
            "recipe": "https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=7.5`, `clips_per_video=5`, and `clip_len=16`"
            ),
            "num_params": 36610672,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 78.477,
                    "acc@5": 93.582,
                }
            },
        },
    )
    DEFAULT = KINETICS400_V1


def mvit_v1_b(*, weights: Optional[MViT_V1_B_Weights] = None, progress: bool = True, **kwargs: Any) -> MViT:
    """
    Constructs a base MViTV1 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    Args:
        weights (:class:`~torchvision.models.video.MViT_V1_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V1_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V1_B_Weights
        :members:
    """
    weights = MViT_V1_B_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "output_channels": [192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768, 768],
        "kernel_q": [[], [3, 3, 3], [], [3, 3, 3], [], [], [], [], [], [], [], [], [], [], [3, 3, 3], []],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=False,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )
