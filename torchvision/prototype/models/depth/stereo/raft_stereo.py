from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.optical_flow.raft as raft
from torch import Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.optical_flow._utils import grid_sample, make_coords_grid, upsample_flow
from torchvision.models.optical_flow.raft import FlowHead, MotionEncoder, ResidualBlock
from torchvision.ops import Conv2dNormActivation
from torchvision.prototype.transforms._presets import StereoMatching
from torchvision.utils import _log_api_usage_once


__all__ = (
    "RaftStereo",
    "raft_stereo_base",
    "raft_stereo_realtime",
    "Raft_Stereo_Base_Weights",
    "Raft_Stereo_Realtime_Weights",
)


class BaseEncoder(raft.FeatureEncoder):
    """Base encoder for FeatureEncoder and ContextEncoder in which weight may be shared.

    See the Raft-Stereo paper section 4.6 on backbone part.
    """

    def __init__(
        self,
        *,
        block: Callable[..., nn.Module] = ResidualBlock,
        layers: Tuple[int, int, int, int] = (64, 64, 96, 128),
        strides: Tuple[int, int, int, int] = (2, 1, 2, 2),
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ):
        # We use layers + (256,) because raft.FeatureEncoder require 5 layers
        # but here we will set the last conv layer to identity
        super().__init__(block=block, layers=layers + (256,), strides=strides, norm_layer=norm_layer)

        # Base encoder don't have the last conv layer of feature encoder
        self.conv = nn.Identity()

        self.output_dim = layers[3]
        num_downsampling = sum([x - 1 for x in strides])
        self.downsampling_ratio = 2 ** (num_downsampling)


class FeatureEncoder(nn.Module):
    """Feature Encoder for Raft-Stereo (see paper section 3.1) that may have shared weight with the Context Encoder.

    The FeatureEncoder takes concatenation of left and right image as input. It produces feature embedding that later
    will be used to construct correlation volume.
    """

    def __init__(
        self,
        base_encoder: BaseEncoder,
        output_dim: int = 256,
        shared_base: bool = False,
        block: Callable[..., nn.Module] = ResidualBlock,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.base_downsampling_ratio = base_encoder.downsampling_ratio
        base_dim = base_encoder.output_dim

        if not shared_base:
            self.residual_block: nn.Module = nn.Identity()
            self.conv = nn.Conv2d(base_dim, output_dim, kernel_size=1)
        else:
            # If we share base encoder weight for Feature and Context Encoder
            # we need to add residual block with InstanceNorm2d and change the kernel size for conv layer
            # see: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/raft_stereo.py#L35-L37
            self.residual_block = block(base_dim, base_dim, norm_layer=nn.InstanceNorm2d, stride=1)
            self.conv = nn.Conv2d(base_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.base_encoder(x)
        x = self.residual_block(x)
        x = self.conv(x)
        return x


class MultiLevelContextEncoder(nn.Module):
    """Context Encoder for Raft-Stereo (see paper section 3.1) that may have shared weight with the Feature Encoder.

    The ContextEncoder takes left image as input, and it outputs concatenated hidden_states and contexts.
    In Raft-Stereo we have multi level GRUs and this context encoder will also multi outputs (list of Tensor)
    that correspond to each GRUs.
    Take note that the length of "out_with_blocks" parameter represent the number of GRU's level.
    args:
        base_encoder (nn.Module): The base encoder part that can have a shared weight with feature_encoder's
            base_encoder because they have same architecture.
        out_with_blocks (List[bool]): The length represent the number of GRU's level (length of output), and
            if the element is True then the output layer on that position will have additional block
        output_dim (int): The dimension of output on each level (default: 256)
        block (Callable[..., nn.Module]): The type of basic block used for downsampling and output layer
            (default: ResidualBlock)
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        out_with_blocks: List[bool],
        output_dim: int = 256,
        block: Callable[..., nn.Module] = ResidualBlock,
    ):
        super().__init__()
        self.num_level = len(out_with_blocks)
        self.base_encoder = base_encoder
        self.base_downsampling_ratio = base_encoder.downsampling_ratio
        base_dim = base_encoder.output_dim

        self.downsample_and_out_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "downsampler": self._make_downsampler(block, base_dim, base_dim) if i > 0 else nn.Identity(),
                        "out_hidden_state": self._make_out_layer(
                            base_dim, output_dim // 2, with_block=out_with_blocks[i], block=block
                        ),
                        "out_context": self._make_out_layer(
                            base_dim, output_dim // 2, with_block=out_with_blocks[i], block=block
                        ),
                    }
                )
                for i in range(self.num_level)
            ]
        )

    def _make_out_layer(self, in_channels, out_channels, with_block=True, block=ResidualBlock):
        layers = []
        if with_block:
            layers.append(block(in_channels, in_channels, norm_layer=nn.BatchNorm2d, stride=1))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def _make_downsampler(self, block, in_channels, out_channels):
        block1 = block(in_channels, out_channels, norm_layer=nn.BatchNorm2d, stride=2)
        block2 = block(out_channels, out_channels, norm_layer=nn.BatchNorm2d, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.base_encoder(x)
        outs = []
        for layer_dict in self.downsample_and_out_layers:
            x = layer_dict["downsampler"](x)
            outs.append(torch.cat([layer_dict["out_hidden_state"](x), layer_dict["out_context"](x)], dim=1))
        return outs


class ConvGRU(raft.ConvGRU):
    """Convolutional Gru unit."""

    # Modified from raft.ConvGRU to accept pre-convolved contexts,
    # see: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/update.py#L23
    def forward(self, h: Tensor, x: Tensor, context: List[Tensor]) -> Tensor:  # type: ignore[override]
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + context[0])
        r = torch.sigmoid(self.convr(hx) + context[1])
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + context[2])
        h = (1 - z) * h + z * q
        return h


class MultiLevelUpdateBlock(nn.Module):
    """The update block which contains the motion encoder and grus

    It must expose a ``hidden_dims`` attribute which is the hidden dimension size of its gru blocks
    """

    def __init__(self, *, motion_encoder: MotionEncoder, hidden_dims: List[int]):
        super().__init__()
        self.motion_encoder = motion_encoder

        # The GRU input size is the size of previous level hidden_dim plus next level hidden_dim
        # if this is the first gru, then we replace previous level with motion_encoder output channels
        # for the last GRU, we don't add the next level hidden_dim
        gru_input_dims = []
        for i in range(len(hidden_dims)):
            input_dim = hidden_dims[i - 1] if i > 0 else motion_encoder.out_channels
            if i < len(hidden_dims) - 1:
                input_dim += hidden_dims[i + 1]
            gru_input_dims.append(input_dim)

        self.grus = nn.ModuleList(
            [
                ConvGRU(input_size=gru_input_dims[i], hidden_size=hidden_dims[i], kernel_size=3, padding=1)
                # Ideally we should reverse the direction during forward to use the gru with the smallest resolution
                # first however currently there is no way to reverse a ModuleList that is jit script compatible
                # hence we reverse the ordering of self.grus on the constructor instead
                # see: https://github.com/pytorch/pytorch/issues/31772
                for i in reversed(list(range(len(hidden_dims))))
            ]
        )

        self.hidden_dims = hidden_dims

    def forward(
        self,
        hidden_states: List[Tensor],
        contexts: List[List[Tensor]],
        corr_features: Tensor,
        disparity: Tensor,
        level_processed: List[bool],
    ) -> List[Tensor]:
        # We call it reverse_i because it has a reversed ordering compared to hidden_states
        # see self.grus on the constructor for more detail
        for reverse_i, gru in enumerate(self.grus):
            i = len(self.grus) - 1 - reverse_i
            if level_processed[i]:
                # X is concatenation of 2x downsampled hidden_dim (or motion_features if no bigger dim) with
                # upsampled hidden_dim (or nothing if not exist).
                if i == 0:
                    features = self.motion_encoder(disparity, corr_features)
                else:
                    # 2x downsampled features from larger hidden states
                    features = F.avg_pool2d(hidden_states[i - 1], kernel_size=3, stride=2, padding=1)

                if i < len(self.grus) - 1:
                    # Concat with 2x upsampled features from smaller hidden states
                    _, _, h, w = hidden_states[i + 1].shape
                    features = torch.cat(
                        [
                            features,
                            F.interpolate(
                                hidden_states[i + 1], size=(2 * h, 2 * w), mode="bilinear", align_corners=True
                            ),
                        ],
                        dim=1,
                    )

                hidden_states[i] = gru(hidden_states[i], features, contexts[i])

                # NOTE: For slow-fast gru, we don't always want to calculate delta disparity for every call on UpdateBlock
                # Hence we move the delta disparity calculation to the RAFT-Stereo main forward

        return hidden_states


class MaskPredictor(raft.MaskPredictor):
    """Mask predictor to be used when upsampling the predicted disparity."""

    # We add out_channels compared to raft.MaskPredictor
    def __init__(self, *, in_channels: int, hidden_size: int, out_channels: int, multiplier: float = 0.25):
        super(raft.MaskPredictor, self).__init__()
        self.convrelu = Conv2dNormActivation(in_channels, hidden_size, norm_layer=None, kernel_size=3)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1, padding=0)
        self.multiplier = multiplier


class CorrPyramid1d(nn.Module):
    """Row-wise correlation pyramid.

    Create a row-wise correlation pyramid with ``num_levels`` level from the outputs of the feature encoder,
    this correlation pyramid will later be used as index to create correlation features using CorrBlock1d.
    """

    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, fmap1: Tensor, fmap2: Tensor) -> List[Tensor]:
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2) on the same row.
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        torch._assert(
            fmap1.shape == fmap2.shape,
            f"Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)",
        )

        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h, w)
        fmap2 = fmap2.view(batch_size, num_channels, h, w)

        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.view(batch_size, h, w, 1, w)
        corr_volume = corr / torch.sqrt(torch.tensor(num_channels, device=corr.device))

        corr_volume = corr_volume.reshape(batch_size * h * w, 1, 1, w)
        corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=(1, 2), stride=(1, 2))
            corr_pyramid.append(corr_volume)

        return corr_pyramid


class CorrBlock1d(nn.Module):
    """The row-wise correlation block.

    Use indexes from correlation pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding row neighbours
    within radius
    """

    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.out_channels = num_levels * (2 * radius + 1)

    def forward(self, centroids_coords: Tensor, corr_pyramid: List[Tensor]) -> Tensor:
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len, device=centroids_coords.device)
        di = di.view(1, 1, neighborhood_side_len, 1).to(centroids_coords.device)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2 but we only use the first one
        # We only consider 1d and take the first dim only
        centroids_coords = centroids_coords[:, :1].permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 1)

        indexed_pyramid = []
        for corr_volume in corr_pyramid:
            x0 = centroids_coords + di  # end shape is (batch_size * h * w, 1, side_len, 1)
            y0 = torch.zeros_like(x0)
            sampling_coords = torch.cat([x0, y0], dim=-1)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                batch_size, h, w, -1
            )
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        expected_output_shape = (batch_size, self.out_channels, h, w)
        torch._assert(
            corr_features.shape == expected_output_shape,
            f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}",
        )
        return corr_features


class RaftStereo(nn.Module):
    def __init__(
        self,
        *,
        feature_encoder: FeatureEncoder,
        context_encoder: MultiLevelContextEncoder,
        corr_pyramid: CorrPyramid1d,
        corr_block: CorrBlock1d,
        update_block: MultiLevelUpdateBlock,
        disparity_head: nn.Module,
        mask_predictor: Optional[nn.Module] = None,
        slow_fast: bool = False,
    ):
        """RAFT-Stereo model from
        `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.

        args:
            feature_encoder (FeatureEncoder): The feature encoder. Its input is the concatenation of ``left_image`` and ``right_image``.
            context_encoder (MultiLevelContextEncoder): The context encoder. Its input is ``left_image``.
                It has multi-level output and each level will have 2 parts:

                - one part will be used as the actual "context", passed to the recurrent unit of the ``update_block``
                - one part will be used to initialize the hidden state of the recurrent unit of
                  the ``update_block``

            corr_pyramid (CorrPyramid1d): Module to build the correlation pyramid from feature encoder output
            corr_block (CorrBlock1d): The correlation block, which uses the correlation pyramid indexes
                to create correlation features. It takes the coordinate of the centroid pixel and correlation pyramid
                as input and returns the correlation features.
                It must expose an ``out_channels`` attribute.

            update_block (MultiLevelUpdateBlock): The update block, which contains the motion encoder, and the recurrent unit.
                It takes as input the hidden state of its recurrent unit, the context, the correlation
                features, and the current predicted disparity. It outputs an updated hidden state
            disparity_head (nn.Module): The disparity head block will convert from the hidden state into changes in disparity.
            mask_predictor (nn.Module, optional): Predicts the mask that will be used to upsample the predicted flow.
                If ``None`` (default), the flow is upsampled using interpolation.
            slow_fast (bool): A boolean that specify whether we should use slow-fast GRU or not. See RAFT-Stereo paper
                on section 3.4 for more detail.
        """
        super().__init__()
        _log_api_usage_once(self)

        # This indicates that the disparity output will be only have 1 channel (represent horizontal axis).
        # We need this because some stereo matching model like CREStereo might have 2 channel on the output
        self.output_channels = 1

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder

        self.base_downsampling_ratio = feature_encoder.base_downsampling_ratio
        self.num_level = self.context_encoder.num_level
        self.corr_pyramid = corr_pyramid
        self.corr_block = corr_block
        self.update_block = update_block
        self.disparity_head = disparity_head
        self.mask_predictor = mask_predictor

        hidden_dims = self.update_block.hidden_dims
        # Follow the original implementation to do pre convolution on the context
        # See: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/raft_stereo.py#L32
        self.context_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dims[i], hidden_dims[i] * 3, kernel_size=3, padding=1) for i in range(self.num_level)]
        )
        self.slow_fast = slow_fast

    def forward(
        self, left_image: Tensor, right_image: Tensor, flow_init: Optional[Tensor] = None, num_iters: int = 12
    ) -> List[Tensor]:
        """
        Return disparity predictions on every iteration as a list of Tensor.
        args:
            left_image (Tensor): The input left image with layout B, C, H, W
            right_image (Tensor): The input right image with layout B, C, H, W
            flow_init (Optional[Tensor]): Initial estimate for the disparity. Default: None
            num_iters (int): Number of update block iteration on the largest resolution. Default: 12
        """
        batch_size, _, h, w = left_image.shape
        torch._assert(
            (h, w) == right_image.shape[-2:],
            f"input images should have the same shape, instead got ({h}, {w}) != {right_image.shape[-2:]}",
        )

        torch._assert(
            (h % self.base_downsampling_ratio == 0 and w % self.base_downsampling_ratio == 0),
            f"input image H and W should be divisible by {self.base_downsampling_ratio}, instead got H={h} and W={w}",
        )

        fmaps = self.feature_encoder(torch.cat([left_image, right_image], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        torch._assert(
            fmap1.shape[-2:] == (h // self.base_downsampling_ratio, w // self.base_downsampling_ratio),
            f"The feature encoder should downsample H and W by {self.base_downsampling_ratio}",
        )

        corr_pyramid = self.corr_pyramid(fmap1, fmap2)

        # Multi level contexts
        context_outs = self.context_encoder(left_image)

        hidden_dims = self.update_block.hidden_dims
        context_out_channels = [context_outs[i].shape[1] - hidden_dims[i] for i in range(len(context_outs))]
        hidden_states: List[Tensor] = []
        contexts: List[List[Tensor]] = []
        for i, context_conv in enumerate(self.context_convs):
            # As in the original paper, the actual output of the context encoder is split in 2 parts:
            # - one part is used to initialize the hidden state of the recurent units of the update block
            # - the rest is the "actual" context.
            hidden_state, context = torch.split(context_outs[i], [hidden_dims[i], context_out_channels[i]], dim=1)
            hidden_states.append(torch.tanh(hidden_state))
            contexts.append(
                # mypy is technically correct here. The return type of `torch.split` was incorrectly annotated with
                # `List[int]` although it should have been `Tuple[Tensor, ...]`. However, the latter is not supported by
                # JIT and thus we have to keep the wrong annotation here and silence mypy.
                torch.split(  # type: ignore[arg-type]
                    context_conv(F.relu(context)), [hidden_dims[i], hidden_dims[i], hidden_dims[i]], dim=1
                )
            )

        _, Cf, Hf, Wf = fmap1.shape
        coords0 = make_coords_grid(batch_size, Hf, Wf).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, Hf, Wf).to(fmap1.device)

        # We use flow_init for cascade inference
        if flow_init is not None:
            coords1 = coords1 + flow_init

        disparity_predictions = []
        for _ in range(num_iters):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = self.corr_block(centroids_coords=coords1, corr_pyramid=corr_pyramid)

            disparity = coords1 - coords0
            if self.slow_fast:
                # Using slow_fast GRU (see paper section 3.4). The lower resolution are processed more often
                for i in range(1, self.num_level):
                    # We only processed the smallest i levels
                    level_processed = [False] * (self.num_level - i) + [True] * i
                    hidden_states = self.update_block(
                        hidden_states, contexts, corr_features, disparity, level_processed=level_processed
                    )
            hidden_states = self.update_block(
                hidden_states, contexts, corr_features, disparity, level_processed=[True] * self.num_level
            )
            # Take the largest hidden_state to get the disparity
            hidden_state = hidden_states[0]
            delta_disparity = self.disparity_head(hidden_state)
            # in stereo mode, project disparity onto epipolar
            delta_disparity[:, 1] = 0.0

            coords1 = coords1 + delta_disparity
            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            upsampled_disparity = upsample_flow(
                (coords1 - coords0), up_mask=up_mask, factor=self.base_downsampling_ratio
            )
            disparity_predictions.append(upsampled_disparity[:, :1])

        return disparity_predictions


def _raft_stereo(
    *,
    weights: Optional[WeightsEnum],
    progress: bool,
    shared_encoder_weight: bool,
    # Feature encoder
    feature_encoder_layers: Tuple[int, int, int, int, int],
    feature_encoder_strides: Tuple[int, int, int, int],
    feature_encoder_block: Callable[..., nn.Module],
    # Context encoder
    context_encoder_layers: Tuple[int, int, int, int, int],
    context_encoder_strides: Tuple[int, int, int, int],
    # if the `out_with_blocks` param of the context_encoder is True, then
    # the particular output on that level position will have additional `context_encoder_block` layer
    context_encoder_out_with_blocks: List[bool],
    context_encoder_block: Callable[..., nn.Module],
    # Correlation block
    corr_num_levels: int,
    corr_radius: int,
    # Motion encoder
    motion_encoder_corr_layers: Tuple[int, int],
    motion_encoder_flow_layers: Tuple[int, int],
    motion_encoder_out_channels: int,
    # Update block
    update_block_hidden_dims: List[int],
    # Flow Head
    flow_head_hidden_size: int,
    # Mask predictor
    mask_predictor_hidden_size: int,
    use_mask_predictor: bool,
    slow_fast: bool,
    **kwargs,
):
    if len(context_encoder_out_with_blocks) != len(update_block_hidden_dims):
        raise ValueError(
            "Length of context_encoder_out_with_blocks and update_block_hidden_dims must be the same"
            + "because both of them represent the number of GRUs level"
        )
    if shared_encoder_weight:
        if (
            feature_encoder_layers[:-1] != context_encoder_layers[:-1]
            or feature_encoder_strides != context_encoder_strides
        ):
            raise ValueError(
                "If shared_encoder_weight is True, then the feature_encoder_layers[:-1]"
                + " and feature_encoder_strides must be the same with context_encoder_layers[:-1] and context_encoder_strides!"
            )

        base_encoder = kwargs.pop("base_encoder", None) or BaseEncoder(
            block=context_encoder_block,
            layers=context_encoder_layers[:-1],
            strides=context_encoder_strides,
            norm_layer=nn.BatchNorm2d,
        )
        feature_base_encoder = base_encoder
        context_base_encoder = base_encoder
    else:
        feature_base_encoder = BaseEncoder(
            block=feature_encoder_block,
            layers=feature_encoder_layers[:-1],
            strides=feature_encoder_strides,
            norm_layer=nn.InstanceNorm2d,
        )
        context_base_encoder = BaseEncoder(
            block=context_encoder_block,
            layers=context_encoder_layers[:-1],
            strides=context_encoder_strides,
            norm_layer=nn.BatchNorm2d,
        )
    feature_encoder = kwargs.pop("feature_encoder", None) or FeatureEncoder(
        feature_base_encoder,
        output_dim=feature_encoder_layers[-1],
        shared_base=shared_encoder_weight,
        block=feature_encoder_block,
    )
    context_encoder = kwargs.pop("context_encoder", None) or MultiLevelContextEncoder(
        context_base_encoder,
        out_with_blocks=context_encoder_out_with_blocks,
        output_dim=context_encoder_layers[-1],
        block=context_encoder_block,
    )

    feature_downsampling_ratio = feature_encoder.base_downsampling_ratio

    corr_pyramid = kwargs.pop("corr_pyramid", None) or CorrPyramid1d(num_levels=corr_num_levels)
    corr_block = kwargs.pop("corr_block", None) or CorrBlock1d(num_levels=corr_num_levels, radius=corr_radius)

    motion_encoder = kwargs.pop("motion_encoder", None) or MotionEncoder(
        in_channels_corr=corr_block.out_channels,
        corr_layers=motion_encoder_corr_layers,
        flow_layers=motion_encoder_flow_layers,
        out_channels=motion_encoder_out_channels,
    )
    update_block = kwargs.pop("update_block", None) or MultiLevelUpdateBlock(
        motion_encoder=motion_encoder, hidden_dims=update_block_hidden_dims
    )

    # We use the largest scale hidden_dims of update_block to get the predicted disparity
    disparity_head = kwargs.pop("disparity_head", None) or FlowHead(
        in_channels=update_block_hidden_dims[0],
        hidden_size=flow_head_hidden_size,
    )

    mask_predictor = kwargs.pop("mask_predictor", None)
    if use_mask_predictor:
        mask_predictor = MaskPredictor(
            in_channels=update_block.hidden_dims[0],
            hidden_size=mask_predictor_hidden_size,
            out_channels=9 * feature_downsampling_ratio * feature_downsampling_ratio,
        )
    else:
        mask_predictor = None

    model = RaftStereo(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_pyramid=corr_pyramid,
        corr_block=corr_block,
        update_block=update_block,
        disparity_head=disparity_head,
        mask_predictor=mask_predictor,
        slow_fast=slow_fast,
        **kwargs,  # not really needed, all params should be consumed by now
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


class Raft_Stereo_Realtime_Weights(WeightsEnum):
    SCENEFLOW_V1 = Weights(
        # Weights ported from https://github.com/princeton-vl/RAFT-Stereo
        url="https://download.pytorch.org/models/raft_stereo_realtime-cf345ccb.pth",
        transforms=partial(StereoMatching, resize_size=(224, 224)),
        meta={
            "num_params": 8077152,
            "recipe": "https://github.com/princeton-vl/RAFT-Stereo",
            "_metrics": {
                # Following metrics from paper: https://arxiv.org/abs/2109.07547
                "Kitty2015": {
                    "3px": 0.9409,
                }
            },
        },
    )

    DEFAULT = SCENEFLOW_V1


class Raft_Stereo_Base_Weights(WeightsEnum):
    SCENEFLOW_V1 = Weights(
        # Weights ported from https://github.com/princeton-vl/RAFT-Stereo
        url="https://download.pytorch.org/models/raft_stereo_base_sceneflow-eff3f2e6.pth",
        transforms=partial(StereoMatching, resize_size=(224, 224)),
        meta={
            "num_params": 11116176,
            "recipe": "https://github.com/princeton-vl/RAFT-Stereo",
            "_metrics": {
                # Following metrics from paper: https://arxiv.org/abs/2109.07547
                # Using standard metrics for each dataset
                "Kitty2015": {
                    # Ratio of pixels with difference less than 3px from ground truth
                    "3px": 0.9426,
                },
                # For middlebury, ratio of pixels with difference less than 2px from ground truth
                # on full, half, and quarter image resolution
                "Middlebury2014-val-full": {
                    "2px": 0.8167,
                },
                "Middlebury2014-val-half": {
                    "2px": 0.8741,
                },
                "Middlebury2014-val-quarter": {
                    "2px": 0.9064,
                },
                "ETH3D-val": {
                    # Ratio of pixels with difference less than 1px from ground truth
                    "1px": 0.9672,
                },
            },
        },
    )

    MIDDLEBURY_V1 = Weights(
        # Weights ported from https://github.com/princeton-vl/RAFT-Stereo
        url="https://download.pytorch.org/models/raft_stereo_base_middlebury-afa9d252.pth",
        transforms=partial(StereoMatching, resize_size=(224, 224)),
        meta={
            "num_params": 11116176,
            "recipe": "https://github.com/princeton-vl/RAFT-Stereo",
            "_metrics": {
                # Following metrics from paper: https://arxiv.org/abs/2109.07547
                "Middlebury-test": {
                    "mae": 1.27,
                    "1px": 0.9063,
                    "2px": 0.9526,
                    "5px": 0.9725,
                }
            },
        },
    )

    ETH3D_V1 = Weights(
        # Weights ported from https://github.com/princeton-vl/RAFT-Stereo
        url="https://download.pytorch.org/models/raft_stereo_base_eth3d-d4830f22.pth",
        transforms=partial(StereoMatching, resize_size=(224, 224)),
        meta={
            "num_params": 11116176,
            "recipe": "https://github.com/princeton-vl/RAFT-Stereo",
            "_metrics": {
                # Following metrics from paper: https://arxiv.org/abs/2109.07547
                "ETH3D-test": {
                    "mae": 0.18,
                    "1px": 0.9756,
                    "2px": 0.9956,
                }
            },
        },
    )

    DEFAULT = MIDDLEBURY_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def raft_stereo_realtime(
    *, weights: Optional[Raft_Stereo_Realtime_Weights] = None, progress=True, **kwargs
) -> RaftStereo:
    """RAFT-Stereo model from
    `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.
    This is the realtime variant of the Raft-Stereo model that is described on the paper section 4.7.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Realtime_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Realtime_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.prototype.models.depth.stereo.raft_stereo.RaftStereo``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.prototype.models.depth.stereo.Raft_Stereo_Realtime_Weights
        :members:
    """

    weights = Raft_Stereo_Realtime_Weights.verify(weights)

    return _raft_stereo(
        weights=weights,
        progress=progress,
        shared_encoder_weight=True,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_strides=(2, 1, 2, 2),
        feature_encoder_block=ResidualBlock,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_strides=(2, 1, 2, 2),
        context_encoder_out_with_blocks=[True, True],
        context_encoder_block=ResidualBlock,
        # Correlation block
        corr_num_levels=4,
        corr_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(64, 64),
        motion_encoder_flow_layers=(64, 64),
        motion_encoder_out_channels=128,
        # Update block
        update_block_hidden_dims=[128, 128],
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        mask_predictor_hidden_size=256,
        use_mask_predictor=True,
        slow_fast=True,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def raft_stereo_base(*, weights: Optional[Raft_Stereo_Base_Weights] = None, progress=True, **kwargs) -> RaftStereo:
    """RAFT-Stereo model from
    `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Base_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Base_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.prototype.models.depth.stereo.raft_stereo.RaftStereo``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.prototype.models.depth.stereo.Raft_Stereo_Base_Weights
        :members:
    """

    weights = Raft_Stereo_Base_Weights.verify(weights)

    return _raft_stereo(
        weights=weights,
        progress=progress,
        shared_encoder_weight=False,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_strides=(1, 1, 2, 2),
        feature_encoder_block=ResidualBlock,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_strides=(1, 1, 2, 2),
        context_encoder_out_with_blocks=[True, True, False],
        context_encoder_block=ResidualBlock,
        # Correlation block
        corr_num_levels=4,
        corr_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(64, 64),
        motion_encoder_flow_layers=(64, 64),
        motion_encoder_out_channels=128,
        # Update block
        update_block_hidden_dims=[128, 128, 128],
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        mask_predictor_hidden_size=256,
        use_mask_predictor=True,
        slow_fast=False,
        **kwargs,
    )
