from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.fx
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t

from ...ops import StochasticDepth, MLP
from .._utils import _make_divisible


__all__ = ["mvit_b_16"]


def _prod(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product


def _attention_pool(
    tensor: torch.Tensor,
    pool: Optional[nn.Module],
    thw_shape: List[int],
    norm: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


    Args:
        tensor (torch.Tensor): Input tensor.
        pool (Optional[Callable]): Pool operation that is applied to the input tensor.
            If pool is none, return the input tensor.
        thw_shape (List): The shape of the input tensor (before flattening).
        norm: (Optional[Callable]): Optional normalization operation applied to
         tensor after pool.

    Returns:
        tensor (torch.Tensor): Input tensor after pool.
        thw_shape (List[int]): Output tensor shape (before flattening).
    """
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor_dim != 4:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    if isinstance(norm, (nn.BatchNorm3d, nn.Identity)):
        # If use BN, we apply norm before pooling instead of after pooling.
        tensor = norm(tensor)
        # We also empirically find that adding a GELU here is beneficial.
        tensor = nn.functional.gelu(tensor)

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None and not isinstance(norm, nn.BatchNorm3d):
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


torch.fx.wrap("_attention_pool")


class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.

    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        depthwise_conv: bool = True,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            depthwise_conv (bool): Wether use depthwise or full convolution for pooling.
            bias_on (bool): Wether use biases for linear layers.
        """

        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True if bias_on else False)
        if dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if kernel_q is not None and _prod(kernel_q) == 1 and _prod(stride_q) == 1:
            kernel_q = None
        if kernel_kv is not None and _prod(kernel_kv) == 1 and _prod(stride_kv) == 1:
            kernel_kv = None

        self.pool_q = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_q is not None
            else None
        )
        self.norm_q = norm_layer(head_dim) if kernel_q is not None else None
        self.pool_k = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_k = norm_layer(head_dim) if kernel_kv is not None else None
        self.pool_v = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_v = norm_layer(head_dim) if kernel_kv is not None else None

    def _qkv_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        thw_shape: List[int],
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]]:
        q, q_shape = _attention_pool(
            q,
            self.pool_q,
            thw_shape,
            norm=self.norm_q,
        )
        k, k_shape = _attention_pool(
            k,
            self.pool_k,
            thw_shape,
            norm=self.norm_k,
        )
        v, v_shape = _attention_pool(
            v,
            self.pool_v,
            thw_shape,
            norm=self.norm_v,
        )
        return q, q_shape, k, k_shape, v, v_shape

    def forward(self, x: torch.Tensor, thw_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        x = (torch.matmul(attn, v) + q).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape



torch.fx.wrap("_attention_pool")

class MultiScaleBlock(nn.Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        droppath_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        depthwise_conv: bool = True,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            depthwise_conv (bool): Wether use depthwise or full convolution for pooling.
            bias_on (bool): Wether use biases for linear layers.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=attn_norm_layer,
            bias_on=bias_on,
            depthwise_conv=depthwise_conv,
        )
        self.drop_path = StochasticDepth(droppath_rate, "row") if droppath_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, [mlp_hidden_dim, dim_out], activation_layer=act_layer, dropout=dropout_rate, bias=bias_on, inplace=None)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out, bias=bias_on)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and _prod(stride_skip) > 1
            else None
        )

    def forward(self, x: torch.Tensor, thw_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        x_block, thw_shape_new = self.attn(
            (
                self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
                if isinstance(self.norm1, nn.BatchNorm1d)
                else self.norm1(x)
            ),
            thw_shape,
        )
        x_res, _ = _attention_pool(x, self.pool_skip, thw_shape)
        x = x_res + self.drop_path(x_block)
        x_norm = (
            self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1) if isinstance(self.norm2, nn.BatchNorm1d) else self.norm2(x)
        )
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
        """
        super().__init__()
        assert len(patch_embed_shape) == 3, "Patch_embed_shape should be in the form of (T, H, W)."
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_spatial_patch, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patch, embed_dim))
        self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed_spatial.repeat(1, self.num_temporal_patch, 1) + torch.repeat_interleave(
            self.pos_embed_temporal,
            self.num_spatial_patch,
            dim=1,
        )
        pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
        x = x + pos_embed

        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        # Projection configs.
        in_features: int,
        out_features: int,
        # Dropout configs.
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pick cls embedding
        x = x[:, 0]
        # Performs dropout.
        x = self.dropout(x)
        # Performs projection.
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Transformer basic patch embedding module. Performs patchifying input, flatten and
    and transpose.

    ::

                                       PatchModel
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    The builder can be found in `create_patch_embed`.

    """

    def __init__(
        self,
        patch_model: nn.Module,
    ) -> None:
        super().__init__()
        self.patch_model = patch_model

    def forward(self, x) -> torch.Tensor:
        x = self.patch_model(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
    Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227

    ::

                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head


    The builder can be found in `create_mvit`.
    """

    def __init__(
        self,
        patch_embed: Optional[nn.Module],
        cls_positional_encoding: nn.Module,
        pos_drop: Optional[nn.Module],
        blocks: nn.ModuleList,
        norm_embed: Optional[nn.Module],
        head: Optional[nn.Module],
    ) -> None:
        """
        Args:
            patch_embed (nn.Module): Patch embed module.
            cls_positional_encoding (nn.Module): Positional encoding module.
            pos_drop (Optional[nn.Module]): Dropout module after patch embed.
            blocks (nn.ModuleList): Stack of multi-scale transformer blocks.
            norm_layer (nn.Module): Normalization layer before head.
            head (Optional[nn.Module]): Head module.
        """
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_positional_encoding = cls_positional_encoding
        self.pos_drop = pos_drop
        self.blocks = blocks
        self.norm_embed = norm_embed
        self.head = head
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, SpatioTemporalClsPositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is not None:
            x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)

        if self.pos_drop is not None:
            x = self.pos_drop(x)

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x


def create_multiscale_vision_transformers(
    spatial_size: _size_2_t,
    temporal_size: int,
    depth: int = 16,
    # Patch embed config.
    input_channels: int = 3,
    patch_embed_dim: int = 96,
    conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
    conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
    conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
    # Attention block config.
    num_heads: int = 1,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    dropout_rate_block: float = 0.0,
    droppath_rate_block: float = 0.0,
    depthwise_conv: bool = True,
    bias_on: bool = True,
    embed_dim_mul: Optional[List[List[int]]] = ([1, 2.0], [3, 2.0], [14, 2.0]),
    atten_head_mul: Optional[List[List[int]]] = ([1, 2.0], [3, 2.0], [14, 2.0]),
    pool_q_stride_size: Optional[List[List[int]]] = ([1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]),
    pool_kv_stride_size: Optional[List[List[int]]] = None,
    pool_kv_stride_adaptive: Optional[_size_3_t] = (1, 8, 8),
    pool_kvq_kernel: Optional[_size_3_t] = (3, 3, 3),
    # Head config.
    head_dropout_rate: float = 0.5,
    num_classes: int = 400,
    **kwargs,
) -> nn.Module:
    """
    Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
    (ViT) is a specific case of MViT that only uses a single scale attention block.

    Args:
        spatial_size (_size_2_t): Input video spatial resolution (H, W). If a single
            int is given, it assumes the width and the height are the same.
        temporal_size (int): Number of frames in the input video.
        depth (int): The depth of the model.

        input_channels (int): Channel dimension of the input video.
        patch_embed_dim (int): Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel (Tuple[int]): Kernel size of the convolution for
            patchifing the video input.
        conv_patch_embed_stride (Tuple[int]): Stride size of the convolution for
            patchifing the video input.
        conv_patch_embed_padding (Tuple[int]): Padding size of the convolution for
            patchifing the video input.

        num_heads (int): Number of heads in the first transformer block.
        mlp_ratio (float): Mlp ratio which controls the feature dimension in the
            hidden layer of the Mlp block.
        qkv_bias (bool): If set to False, the qkv layer will not learn an additive
            bias. Default: True.
        dropout_rate_block (float): Dropout rate for the attention block.
        droppath_rate_block (float): Droppath rate for the attention block.
        depthwise_conv (bool): Wether use depthwise or full convolution for pooling.
        bias_on (bool): Wether use biases for linear layers.
        embed_dim_mul (Optional[List[List[int]]]): Dimension multiplication at layer i.
            If X is used, then the next block will increase the embed dimension by X
            times. Format: [depth_i, mul_dim_ratio].
        atten_head_mul (Optional[List[List[int]]]): Head dimension multiplication at
            layer i. If X is used, then the next block will increase the head by
            X times. Format: [depth_i, mul_dim_ratio].
        pool_q_stride_size (Optional[List[List[int]]]): List of stride sizes for the
            pool q at each layer. Format:
            [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_size (Optional[List[List[int]]]): List of stride sizes for the
            pool kv at each layer. Format:
            [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_adaptive (Optional[_size_3_t]): Initial kv stride size for the
            first block. The stride size will be further reduced at the layer where q
            is pooled with the ratio of the stride of q pooling. If
            pool_kv_stride_adaptive is set, then pool_kv_stride_size should be none.
        pool_kvq_kernel (Optional[_size_3_t]): Pooling kernel size for q and kv. It None,
            the kernel_size is [s + 1 if s > 1 else s for s in stride_size].

        head_dropout_rate (float): Dropout rate in the head.
        num_classes (int): Number of classes in the final classification head.

    Example usage (building a MViT_B model for Kinetics400):

        spatial_size = 224
        temporal_size = 16
        embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        pool_kv_stride_adaptive = [1, 8, 8]
        pool_kvq_kernel = [3, 3, 3]
        num_classes = 400
        MViT_B = create_multiscale_vision_transformers(
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            embed_dim_mul=embed_dim_mul,
            atten_head_mul=atten_head_mul,
            pool_q_stride_size=pool_q_stride_size,
            pool_kv_stride_adaptive=pool_kv_stride_adaptive,
            pool_kvq_kernel=pool_kvq_kernel,
            num_classes=num_classes,
        )
    """

    if pool_kv_stride_adaptive is not None:
        assert pool_kv_stride_size is None, "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    block_norm_layer = partial(nn.LayerNorm, eps=1e-6)
    attn_norm_layer = partial(nn.LayerNorm, eps=1e-6)

    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    patch_embed = PatchEmbed(
        patch_model=nn.Conv3d(
            in_channels=input_channels,
            out_channels=patch_embed_dim,
            kernel_size=conv_patch_embed_kernel,
            stride=conv_patch_embed_stride,
            padding=conv_patch_embed_padding,
            bias=True,
        )
    )

    input_dims = (temporal_size, spatial_size[0], spatial_size[1])

    patch_embed_shape = tuple(v // conv_patch_embed_stride[i] for i, v in enumerate(input_dims))

    cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_shape,
    )

    dpr = [x.item() for x in torch.linspace(0, droppath_rate_block, depth)]  # stochastic depth decay rule

    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

    mvit_blocks = nn.ModuleList()

    pool_q = [[] for _ in range(depth)]
    pool_kv = [[] for _ in range(depth)]
    stride_q = [[] for _ in range(depth)]
    stride_kv = [[] for _ in range(depth)]

    if pool_q_stride_size is not None:
        for i in range(len(pool_q_stride_size)):
            stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride_size[i][0]] = [s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]]

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if pool_kv_stride_adaptive is not None:
        _stride_kv = pool_kv_stride_adaptive
        pool_kv_stride_size = []
        for i in range(depth):
            if len(stride_q[i]) > 0:
                _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
            pool_kv_stride_size.append([i] + list(_stride_kv))

    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]]

    dim_out = 0
    for i in range(depth):
        num_heads = _make_divisible(num_heads * head_mul[i], 1)
        patch_embed_dim = _make_divisible(patch_embed_dim * dim_mul[i], num_heads, min_value=8)
        dim_out = _make_divisible(
            patch_embed_dim * dim_mul[i + 1], divisor=_make_divisible(num_heads * head_mul[i + 1], 8), min_value=8,
        )

        mvit_blocks.append(
            MultiScaleBlock(
                dim=patch_embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate_block,
                droppath_rate=dpr[i],
                norm_layer=block_norm_layer,
                attn_norm_layer=attn_norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                bias_on=bias_on,
                depthwise_conv=depthwise_conv,
            )
        )

    embed_dim = dim_out
    head_model = ClassificationHead(
        in_features=embed_dim,
        out_features=num_classes,
        dropout_rate=head_dropout_rate,
    )

    return MultiscaleVisionTransformers(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=nn.Dropout(p=dropout_rate_block) if dropout_rate_block > 0.0 else None,
        blocks=mvit_blocks,
        norm_embed=norm_layer(embed_dim),
        head=head_model,
    )


def mvit_b_16(
    spatial_size=224,
    temporal_size=16,
    num_classes=400,
    **kwargs,
):
    return create_multiscale_vision_transformers(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        num_classes=num_classes,
        **kwargs,
    )
