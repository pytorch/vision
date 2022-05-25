from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.fx
import torch.nn as nn

from ...ops import StochasticDepth, MLP
from .._utils import _make_divisible


__all__ = ["mvit_b_16"]


def _prod(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product


def _unsqueeze(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    tensor_dim = tensor.dim()
    if tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor_dim != 4:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor, tensor_dim


def _squeeze(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor


torch.fx.wrap("_unsqueeze")
torch.fx.wrap("_squeeze")


class AttentionPool(nn.Module):
    def __init__(self, pool: Optional[nn.Module], norm: Optional[nn.Module]):
        super().__init__()
        self.pool = pool
        self.norm = norm
        # The standard mvit uses layer norm and normalizes after pooling. Nevertheless in some production use-cases, it
        # might be prefered to "absorb" the norm in order to make the inference faster.
        self.norm_before_pool = isinstance(norm, (nn.BatchNorm3d, nn.Identity))

    def forward(
        self,
        tensor: torch.Tensor,
        thw_shape: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        if self.pool is None:
            return tensor, thw_shape
        tensor, tensor_dim = _unsqueeze(tensor)

        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

        B, N, L, C = tensor.shape
        T, H, W = thw_shape
        tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        if self.norm is not None and self.norm_before_pool:
            # If use BN, we apply norm before pooling instead of after pooling.
            tensor = self.norm(tensor)
            # We also empirically find that adding a GELU here is beneficial.
            tensor = nn.functional.gelu(tensor)

        tensor = self.pool(tensor)

        thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
        L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
        tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

        tensor = torch.cat((cls_tok, tensor), dim=2)
        if self.norm is not None and not self.norm_before_pool:
            tensor = self.norm(tensor)

        tensor = _squeeze(tensor, tensor_dim)
        return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        kernel_q: Tuple[int, int, int] = (1, 1, 1),
        kernel_kv: Tuple[int, int, int] = (1, 1, 1),
        stride_q: Tuple[int, int, int] = (1, 1, 1),
        stride_kv: Tuple[int, int, int] = (1, 1, 1),
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            dropout_rate (float): Dropout rate.
            kernel_q (Tuple[int, int, int]): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (Tuple[int, int, int]): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (Tuple[int, int, int]): Pooling kernel stride for q.
            stride_kv (Tuple[int, int, int]): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
        """

        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        skip_pool_q = _prod(kernel_q) == 1 and _prod(stride_q) == 1
        skip_pool_kv = _prod(kernel_kv) == 1 and _prod(stride_kv) == 1

        self.att_pool_q = AttentionPool(
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=head_dim,
                bias=False,
            )
            if not skip_pool_q
            else None,
            norm_layer(head_dim) if not skip_pool_q else None,
        )
        self.att_pool_k = AttentionPool(
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim,
                bias=False,
            )
            if not skip_pool_kv
            else None,
            norm_layer(head_dim) if not skip_pool_kv else None,
        )
        self.att_pool_v = AttentionPool(
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim,
                bias=False,
            )
            if not skip_pool_kv
            else None,
            norm_layer(head_dim) if not skip_pool_kv else None,
        )

    def forward(self, x: torch.Tensor, thw_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, q_shape = self.att_pool_q(
            q,
            thw_shape,
        )
        k, k_shape = self.att_pool_k(
            k,
            thw_shape,
        )
        v, v_shape = self.att_pool_v(
            v,
            thw_shape,
        )

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        x = (torch.matmul(attn, v) + q).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        kernel_q: Tuple[int, int, int] = (1, 1, 1),
        kernel_kv: Tuple[int, int, int] = (1, 1, 1),
        stride_q: Tuple[int, int, int] = (1, 1, 1),
        stride_kv: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            stochastic_depth_prob (float): Stochastic Depth probability. If set to 0, it's disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (Tuple[int, int, int]): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (Tuple[int, int, int]): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (Tuple[int, int, int]): Pooling kernel stride for q.
            stride_kv (Tuple[int, int, int]): Pooling kernel stride for kv.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
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
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=attn_norm_layer,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = 4 * dim  # 4x mlp ratio
        self.mlp = MLP(
            dim, [mlp_hidden_dim, dim_out], activation_layer=act_layer, dropout=dropout_rate, inplace=None
        )
        self.proj: Optional[nn.Module] = None
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.att_pool_skip = AttentionPool(
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and _prod(stride_skip) > 1
            else None,
            None,
        )
        self.need_permutation = [isinstance(self.norm1, nn.BatchNorm1d), isinstance(self.norm2, nn.BatchNorm1d)]

    def forward(self, x: torch.Tensor, thw_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        x_block, thw_shape_new = self.attn(
            (self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1) if self.need_permutation[0] else self.norm1(x)),
            thw_shape,
        )
        x_res, _ = self.att_pool_skip(x, thw_shape)
        x = x_res + self.stochastic_depth(x_block)
        x_norm = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1) if self.need_permutation[1] else self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.proj is not None:
            x = self.proj(x_norm)
        x = x + self.stochastic_depth(x_mlp)
        return x, thw_shape_new


class SpatioTemporalClsPositionalEncoding(nn.Module):
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
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        x = self.dropout(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
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
    """

    def __init__(
        self,
        patch_embed: Optional[nn.Module],
        cls_positional_encoding: nn.Module,
        blocks: nn.ModuleList,
        norm_embed: Optional[nn.Module],
        head: Optional[nn.Module],
    ) -> None:
        """
        Args:
            patch_embed (nn.Module): Patch embed module.
            cls_positional_encoding (nn.Module): Positional encoding module.
            blocks (nn.ModuleList): Stack of multi-scale transformer blocks.
            norm_layer (nn.Module): Normalization layer before head.
            head (Optional[nn.Module]): Head module.
        """
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_positional_encoding = cls_positional_encoding
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

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x


def create_multiscale_vision_transformers(
    spatial_size: Tuple[int, int],
    temporal_size: int,
    depth: int = 16,
    # Patch embed config.
    patch_embed_dim: int = 96,
    conv_patch_embed_kernel: Tuple[int, int, int] = (3, 7, 7),
    conv_patch_embed_stride: Tuple[int, int, int] = (2, 4, 4),
    conv_patch_embed_padding: Tuple[int, int, int] = (1, 3, 3),
    # Attention block config.
    num_heads: int = 1,
    stochastic_depth_prob_block: float = 0.0,
    embed_dim_mul: Optional[List[List[int]]] = ([1, 2.0], [3, 2.0], [14, 2.0]),
    atten_head_mul: Optional[List[List[int]]] = ([1, 2.0], [3, 2.0], [14, 2.0]),
    pool_q_stride_size: Optional[List[List[int]]] = ([1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]),
    pool_kv_stride_size: Optional[List[List[int]]] = None,
    pool_kv_stride_adaptive: Optional[Tuple[int, int, int]] = (1, 8, 8),
    pool_kvq_kernel: Optional[Tuple[int, int, int]] = (3, 3, 3),
    # Head config.
    head_dropout_rate: float = 0.5,
    num_classes: int = 400,
    **kwargs,
) -> nn.Module:
    """
    Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
    (ViT) is a specific case of MViT that only uses a single scale attention block.

    Args:
        spatial_size (Tuple[int, int]): Input video spatial resolution (H, W). If a single
            int is given, it assumes the width and the height are the same.
        temporal_size (int): Number of frames in the input video.
        depth (int): The depth of the model.

        patch_embed_dim (int): Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel (Tuple[int]): Kernel size of the convolution for
            patchifing the video input.
        conv_patch_embed_stride (Tuple[int]): Stride size of the convolution for
            patchifing the video input.
        conv_patch_embed_padding (Tuple[int]): Padding size of the convolution for
            patchifing the video input.

        num_heads (int): Number of heads in the first transformer block.
        stochastic_depth_prob_block (float): Stochastic Depth probability for the attention block.
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
        pool_kv_stride_adaptive (Optional[Tuple[int, int, int]]): Initial kv stride size for the
            first block. The stride size will be further reduced at the layer where q
            is pooled with the ratio of the stride of q pooling. If
            pool_kv_stride_adaptive is set, then pool_kv_stride_size should be none.
        pool_kvq_kernel (Optional[Tuple[int, int, int]]): Pooling kernel size for q and kv. It None,
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
            in_channels=3,
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

    dpr = [x.item() for x in torch.linspace(0, stochastic_depth_prob_block, depth)]  # stochastic depth decay rule

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
            patch_embed_dim * dim_mul[i + 1],
            divisor=_make_divisible(num_heads * head_mul[i + 1], 8),
            min_value=8,
        )

        mvit_blocks.append(
            MultiScaleBlock(
                dim=patch_embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                stochastic_depth_prob=dpr[i],
                norm_layer=block_norm_layer,
                attn_norm_layer=attn_norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
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
