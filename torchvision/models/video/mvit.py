import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t
from torchvision.ops import StochasticDepth


__all__ = ["create_mvit_b_16", "create_multiscale_vision_transformers"]


class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.

    ::

                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        dropout_rate: float = 0.0,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_on)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_on)
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        return x


def _attention_pool(
    tensor: torch.Tensor,
    pool: Optional[Callable],
    thw_shape: List[int],
    norm: Optional[Callable] = None,
) -> torch.Tensor:
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
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
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

    if tensor_dim == 4:
        pass
    else:  # For the case tensor_dim == 3.
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


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
        norm_layer: Callable = nn.LayerNorm,
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
        if kernel_q is not None and numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = None
        if kernel_kv is not None and numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
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

    def _qkv_proj(
        self,
        q: torch.Tensor,
        q_size: List[int],
        k: torch.Tensor,
        k_size: List[int],
        v: torch.Tensor,
        v_size: List[int],
        batch_size: List[int],
        chan_size: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(q).reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        return q, k, v

    def _qkv_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        thw_shape: Tuple[torch.Tensor, List[int]],
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]]:
        q, q_shape = _attention_pool(
            q,
            self.pool_q,
            thw_shape,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = _attention_pool(
            k,
            self.pool_k,
            thw_shape,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = _attention_pool(
            v,
            self.pool_v,
            thw_shape,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int]:
        q_N = numpy.prod(q_shape) + 1
        k_N = numpy.prod(k_shape) + 1
        v_N = numpy.prod(v_shape) + 1
        return q_N, k_N, v_N

    def _reshape_qkv_to_seq(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_N: int,
        v_N: int,
        k_N: int,
        B: int,
        C: int,
    ) -> Tuple[int]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

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

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        x = (attn @ v + q).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


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
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_norm_layer: nn.Module = nn.LayerNorm,
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
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
            bias_on=bias_on,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out, bias=bias_on)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
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


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def set_attributes(self, params: List[object] = None) -> None:
    """
    An utility function used in classes to set attributes from the input list of parameters.
    Args:
        params (list): list of parameters.
    """
    if params:
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
        ceil (bool): If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def round_repeats(repeats, multiplier):
    """
    Round number of layers based on depth multiplier.
    """
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SequencePool(nn.Module):
    """
    Sequence pool produces a single embedding from a sequence of embeddings. Currently
    it supports "mean" and "cls".

    """

    def __init__(self) -> None:

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        return x


class create_vit_basic_head(nn.Module):
    def __init__(
        self,
        # Projection configs.
        in_features: int,
        out_features: int,
        # Pooling configs.
        seq_pool_type: str = "cls",
        # Dropout configs.
        dropout_rate: float = 0.5,
        # Activation configs.
        activation: Callable = None,
    ) -> nn.Module:
        """
        Creates vision transformer basic head.

        ::


                                            Pooling
                                              ↓
                                            Dropout
                                              ↓
                                          Projection
                                              ↓
                                          Activation


        Activation examples include: ReLU, Softmax, Sigmoid, and None.
        Pool type examples include: cls, mean and none.

        Args:

            in_features: input channel size of the resnet head.
            out_features: output channel size of the resnet head.

            pool_type (str): Pooling type. It supports "cls", "mean " and "none". If set to
                "cls", it assumes the first element in the input is the cls token and
                returns it. If set to "mean", it returns the mean of the entire sequence.

            activation (callable): a callable that constructs vision transformer head
                activation layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and
                None (not applying activation).

            dropout_rate (float): dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pick cls embedding
        x = x[:, 0]
        # Performs dropout.
        if self.dropout is not None:
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
        patch_model: nn.Module = None,
    ) -> None:
        super().__init__()
        set_attributes(self, locals())
        assert self.patch_model is not None

    def forward(self, x) -> torch.Tensor:
        x = self.patch_model(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


def create_conv_patch_embed(
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (1, 16, 16),
    conv_stride: Tuple[int] = (1, 4, 4),
    conv_padding: Tuple[int] = (1, 7, 7),
    conv_bias: bool = True,
) -> nn.Module:
    """
    Creates the transformer basic patch embedding. It performs Convolution, flatten and
    transpose.

    ::

                                        Conv3d
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.
        conv (callable): Callable used to build the convolution layer.

    Returns:
        (nn.Module): transformer patch embedding layer.
    """
    conv_module = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,
    )
    return PatchEmbed(patch_model=conv_module)


def _init_resnet_weights(model: nn.Module, fc_init_std: float = 0.01) -> None:
    """
    Performs ResNet style weight initialization. That is, recursively initialize the
    given model in the following way for each type:
        Conv - Follow the initialization of kaiming_normal:
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
        BatchNorm - Set weight and bias of last BatchNorm at every residual bottleneck
            to 0.
        Linear - Set weight to 0 mean Gaussian with std deviation fc_init_std and bias
            to 0.
    Args:
        model (nn.Module): Model to be initialized.
        fc_init_std (float): the expected standard deviation for fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            if m.weight is not None:
                if hasattr(m, "block_final_bn") and m.block_final_bn:
                    m.weight.data.fill_(0.0)
                else:
                    m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            if hasattr(m, "xavier_init") and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def _init_vit_weights(model: nn.Module, trunc_normal_std: float = 0.02) -> None:
    """
    Weight initialization for vision transformers.

    Args:
        model (nn.Module): Model to be initialized.
        trunc_normal_std (float): the expected standard deviation for fully-connected
            layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)


def init_net_weights(
    model: nn.Module,
    init_std: float = 0.01,
    style: str = "resnet",
) -> None:
    """
    Performs weight initialization. Options include ResNet style weight initialization
    and transformer style weight initialization.

    Args:
        model (nn.Module): Model to be initialized.
        init_std (float): The expected standard deviation for initialization.
        style (str): Options include "resnet" and "vit".
    """
    assert style in ["resnet", "vit"]
    if style == "resnet":
        return _init_resnet_weights(model, init_std)
    elif style == "vit":
        return _init_vit_weights(model, init_std)
    else:
        raise NotImplementedError


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
        set_attributes(self, locals())
        assert hasattr(
            cls_positional_encoding, "patch_embed_shape"
        ), "cls_positional_encoding should have attribute patch_embed_shape."
        init_net_weights(self, init_std=0.02, style="vit")

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
    norm: str = "layernorm",
    # Patch embed config.
    input_channels: int = 3,
    patch_embed_dim: int = 96,
    conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
    conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
    conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
    enable_patch_embed_norm: bool = False,
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
    head: Optional[Callable] = create_vit_basic_head,
    head_dropout_rate: float = 0.5,
    head_activation: Callable = None,
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
        norm (str): Normalization layer. It currently supports "layernorm".

        input_channels (int): Channel dimension of the input video.
        patch_embed_dim (int): Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel (Tuple[int]): Kernel size of the convolution for
            patchifing the video input.
        conv_patch_embed_stride (Tuple[int]): Stride size of the convolution for
            patchifing the video input.
        conv_patch_embed_padding (Tuple[int]): Padding size of the convolution for
            patchifing the video input.
        enable_patch_embed_norm (bool): If True, apply normalization after patchifing
            the video input.

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

        head (Callable): Head model.
        head_dropout_rate (float): Dropout rate in the head.
        head_activation (Callable): Activation in the head.
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

    patch_embed = create_conv_patch_embed(
        in_channels=input_channels,
        out_channels=patch_embed_dim,
        conv_kernel_size=conv_patch_embed_kernel,
        conv_stride=conv_patch_embed_stride,
        conv_padding=conv_patch_embed_padding,
    )

    input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
    input_stirde = conv_patch_embed_stride

    patch_embed_shape = [input_dims[i] // input_stirde[i] for i in range(len(input_dims))]

    cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_shape,
    )

    dpr = [x.item() for x in torch.linspace(0, droppath_rate_block, depth)]  # stochastic depth decay rule

    if dropout_rate_block > 0.0:
        pos_drop = nn.Dropout(p=dropout_rate_block)

    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

    mvit_blocks = nn.ModuleList()

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

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
            pool_kv_stride_size.append([i] + _stride_kv)

    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]]

    for i in range(depth):
        num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
        patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
        dim_out = round_width(
            patch_embed_dim,
            dim_mul[i + 1],
            divisor=round_width(num_heads, head_mul[i + 1]),
        )

        block_func = MultiScaleBlock

        mvit_blocks.append(
            block_func(
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
    norm_embed = None if norm_layer is None else norm_layer(embed_dim)
    if head is not None:
        head_model = head(
            in_features=embed_dim,
            out_features=num_classes,
            dropout_rate=head_dropout_rate,
            activation=head_activation,
        )
    else:
        head_model = None

    return MultiscaleVisionTransformers(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
        blocks=mvit_blocks,
        norm_embed=norm_embed,
        head=head_model,
    )


def create_mvit_b_16(
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
