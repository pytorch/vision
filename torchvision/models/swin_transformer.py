from functools import partial
from typing import Optional, Callable, List, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .._internally_replaced_utils import load_state_dict_from_url
from ..ops.stochastic_depth import StochasticDepth
from .convnext import Permute
from .vision_transformer import MLPBlock


__all__ = [
    "SwinTransformer",
    "swin_tiny",
]


_MODELS_URLS = {
    "swin_tiny": "",
    "swin_base": "",
}


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        B, H, W, C = x.shape
        # assert H % 2 == 0 and W % 2 == 0, f"input size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, 2 * C)
        return x


def generate_attention_mask(x: Tensor, window_size: int, shift_size: int):
    """Generate shifted window attention mask"""
    mask = x.new_zeros((1, x.size(1), x.size(2), 1))
    slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, None))
    count = 0
    for h in slices:
        for w in slices:
            mask[:, h[0] : h[1], w[0] : w[1], :] = count
            count += 1
    return mask


torch.fx.wrap("generate_attention_mask")


class ShiftedWindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (int)): The size of the window.
        shift_size (int): Shift size for SW-MSA.
        num_heads (int): Number of attention heads.
        qkv_bias (bool):  If True, add a learnable bias to query, key, value. Default: True.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor):
        B, H, W, C = x.shape
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = self.partition_window(x)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, int(self.window_size ** 2), C)  # nW*B, window_size*window_size, C

        # multi-head attention
        qkv = (
            self.qkv(x_windows)
            .reshape(x_windows.size(0), x_windows.size(1), 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index].view(
            int(self.window_size ** 2), int(self.window_size ** 2), -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.shift_size > 0:
            # generate attention mask
            attn_mask = generate_attention_mask(x, self.window_size, self.shift_size)
            attn_mask = self.partition_window(attn_mask)
            attn_mask = attn_mask.view(-1, int(self.window_size ** 2))
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            num_windows = attn_mask.shape[0]
            attn = attn.view(
                x_windows.size(0) // num_windows, num_windows, self.num_heads, x_windows.size(1), x_windows.size(1)
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x_windows.size(1), x_windows.size(1))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        x_windows = (attn @ v).transpose(1, 2).reshape(x_windows.size(0), x_windows.size(1), C)
        x_windows = self.proj(x_windows)
        x_windows = self.dropout(x_windows)

        # reverse windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = self.reverse_window(x_windows, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, Hp, Wp, C)
        x = x[:, :H, :W, :].contiguous()

        return x

    def partition_window(self, x: Tensor):
        """
        Partition the input tensor into windows: (B, H, W, C) -> (B*nW, window_size, window_size, C).
        """
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size, self.window_size, C)
        return x

    def reverse_window(self, x: Tensor, height: int, width: int):
        """
        The Inverse operation of `partition_window`.
        """
        B = x.shape[0] // ((height * width) // int((self.window_size ** 2)))
        x = x.view(B, height // self.window_size, width // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, height, width, -1)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = ShiftedWindowAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)

        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (int): Patch size. Default: 4.
        num_classes (int): Number of classes for classification head. Default: 1000.
        embed_dim (int): Patch embedding dimension. Default: 96.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7.
        shift_size (List(int)): Shift size of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        dropout (float): Dropout rate. Default: 0.
        attention_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        block (nn.Module): SwinTransformer Block.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(
        self,
        patch_size: int = 4,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        shift_sizes: List[int] = [3, 3, 3, 0],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=shift_sizes[i_stage] if i_layer % 2 == 0 else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(
                    # nn.Sequential(
                    #     norm_layer(dim),
                    #     Permute([0, 3, 1, 2]),
                    #     nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False),
                    #     Permute([0, 2, 3, 1]),
                    # )
                    PatchMerging(dim, norm_layer)
                )

        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def _swin_transformer(
    arch: str,
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: int,
    shift_sizes: List[int],
    stochastic_depth_prob: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    model = SwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )
    if pretrained:
        if arch not in _MODELS_URLS:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(_MODELS_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def swin_tiny(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    `"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _swin_transformer(
        arch="swin_tiny",
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        shift_sizes=[3, 3, 3, 0],
        stochastic_depth_prob=0.2,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )
