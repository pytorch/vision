from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.optical_flow.raft as raft
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models.optical_flow._utils import make_coords_grid
from torchvision.models.optical_flow.raft import ResidualBlock, MotionEncoder, FlowHead
from torchvision.ops import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once


def grid_sample(img: Tensor, absolute_grid: Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    # Add condition h > 1 to prevent nan when h == 1 (in raft-stereo we always have h == 1 because we only want to correlate on the same row)
    # See: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/utils.py#L64
    if h > 1:
        ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)


def upsample_with_mask_and_factor(x, up_mask: Optional[Tensor] = None, factor=8):
    """Upsample tensor x to have factor times the size

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights.
    """
    batch_size, num_channels, h, w = x.shape
    new_h, new_w = h * factor, w * factor

    if up_mask is None:
        return factor * F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    up_mask = up_mask.view(batch_size, 1, 9, factor, factor, h, w)
    up_mask = torch.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

    upsampled_x = F.unfold(factor * x, kernel_size=3, padding=1).view(batch_size, num_channels, 9, 1, 1, h, w)
    upsampled_x = torch.sum(up_mask * upsampled_x, dim=2)

    return upsampled_x.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, num_channels, new_h, new_w)


class BaseEncoder(nn.Module):
    """The feature encoder, used both as the actual feature encoder, and as the context encoder.

    It must downsample its input by 8.
    """

    def __init__(
        self, *, block=ResidualBlock, layers=(64, 64, 96, 128), strides=(2, 1, 2, 2), norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        if len(layers) != 4:
            raise ValueError(f"The expected number of layers is 4, instead got {len(layers)}")

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu = Conv2dNormActivation(
            3, layers[0], norm_layer=norm_layer, kernel_size=7, stride=strides[0], bias=True
        )

        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_layer=norm_layer, first_stride=strides[1])
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_layer=norm_layer, first_stride=strides[2])
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_layer=norm_layer, first_stride=strides[3])
        self.output_dim = layers[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        num_downsampling = sum([x - 1 for x in strides])
        self.downsampling_ratio = 2 ** (num_downsampling)

    def _make_2_blocks(self, block, in_channels, out_channels, norm_layer, first_stride):
        block1 = block(in_channels, out_channels, norm_layer=norm_layer, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x = self.convnormrelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, base_encoder, output_dim=256, shared_base=False, block=ResidualBlock):
        super().__init__()
        self.base_encoder = base_encoder
        self.base_downsampling_ratio = base_encoder.downsampling_ratio
        base_dim = base_encoder.output_dim

        # If we share base encoder weight for Feature and Context Encoder
        # we need to add residual block with InstanceNorm2d and change the kernel size for conv layer
        # see: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/raft_stereo.py#L35-L37
        self.shared_base = shared_base
        if shared_base:
            self.residual_block = block(base_dim, base_dim, norm_layer=nn.InstanceNorm2d, stride=1)
            self.conv = nn.Conv2d(base_dim, output_dim, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(base_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.base_encoder(x)
        if self.shared_base:
            x = self.residual_block(x)
        x = self.conv(x)
        return x


class MultiLevelContextEncoder(nn.Module):
    def __init__(self, base_encoder, output_dim=256, out_with_blocks=[1, 1, 0], block=ResidualBlock):
        super().__init__()
        self.num_level = len(out_with_blocks)
        self.base_encoder = base_encoder
        self.base_downsampling_ratio = base_encoder.downsampling_ratio
        base_dim = base_encoder.output_dim

        # Layer to output hidden_state and context separately (each produce output_dim//2 dims)
        self.out_hidden_states = nn.ModuleList(
            [
                self._make_out_layer(base_dim, output_dim // 2, with_block=with_block, block=block)
                for with_block in out_with_blocks
            ]
        )
        self.out_contexts = nn.ModuleList(
            [
                self._make_out_layer(base_dim, output_dim // 2, with_block=with_block, block=block)
                for with_block in out_with_blocks
            ]
        )

        self.downsamplers = nn.ModuleList(
            [self._make_downsampler(block, base_dim, base_dim) for i in range(1, self.num_level)]
        )

    def _make_out_layer(self, in_channels, out_channels, with_block=1, block=ResidualBlock):
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if with_block:
            block_layer = block(in_channels, in_channels, norm_layer=nn.BatchNorm2d, stride=1)
            return nn.Sequential(block_layer, conv_layer)
        else:
            return conv_layer

    def _make_downsampler(self, block, in_channels, out_channels):
        block1 = block(in_channels, out_channels, norm_layer=nn.BatchNorm2d, stride=2)
        block2 = block(out_channels, out_channels, norm_layer=nn.BatchNorm2d, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x = self.base_encoder(x)
        outs = [torch.cat([self.out_hidden_states[0](x), self.out_contexts[0](x)], dim=1)]
        for i in range(0, self.num_level - 1):
            x = self.downsamplers[i](x)
            outs.append(torch.cat([self.out_hidden_states[i + 1](x), self.out_contexts[i + 1](x)], dim=1))
        return outs


# Modified torchvision.models.optical_flow.raft.ConvGRU
class ConvGRU(raft.ConvGRU):
    """Convolutional Gru unit."""

    # Modified to accept pre-convolved contexts, see: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/update.py#L23
    def forward(self, h, context, x):
        hx = torch.cat([h, x], dim=1)
        cz, cr, cq = context
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q
        return h


class MultiLevelUpdateBlock(nn.Module):
    """The update block which contains the motion encoder and grus

    It must expose a ``hidden_dims`` attribute which is the hidden dimension size of its gru blocks
    """

    def __init__(self, *, motion_encoder, hidden_dims=[128, 128, 128]):
        super().__init__()
        self.motion_encoder = motion_encoder

        # The GRU input size is the size of previous level hidden_dim plus next level hidden_dim
        # if this is the first gru, then we replace previous level with motion_encoder output channels
        # for the last GRU, we dont add the next level hidden_dim
        gru_input_dims = []
        for i in range(len(hidden_dims)):
            input_dim = hidden_dims[i - 1] if i > 0 else motion_encoder.out_channels
            if i < len(hidden_dims) - 1:
                input_dim += hidden_dims[i + 1]
            gru_input_dims.append(input_dim)

        self.grus = nn.ModuleList(
            [
                ConvGRU(input_size=gru_input_dims[i], hidden_size=hidden_dims[i], kernel_size=3, padding=1)
                for i in range(len(hidden_dims))
            ]
        )

        self.hidden_dims = hidden_dims

    def _downsample2x(self, x):
        return F.avg_pool2d(x, 3, stride=2, padding=1)

    def _upsample2x(self, x):
        _, _, h, w = x.shape
        new_h, new_w = h * 2, w * 2
        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    def forward(self, hidden_states, contexts, corr_features, depth, level_processed=[1, 1, 1]):
        for i in range(len(self.grus) - 1, -1, -1):
            if level_processed[i]:
                # X is concatination of downsampled hidden_dim (or motion_features if no bigger dim) with
                # upsampled hidden_dim (or nothing if not exist)
                features = (
                    self._downsample2x(hidden_states[i - 1]) if i > 0 else self.motion_encoder(depth, corr_features)
                )
                if i < len(self.grus) - 1:
                    features = torch.cat([features, self._upsample2x(hidden_states[i + 1])], dim=1)

                hidden_states[i] = self.grus[i](hidden_states[i], contexts[i], features)

                # NOTE: For slow-fast gru, we dont always want to calculate delta depth for every call on UpdateBlock
                # Hence we move the delta depth calculation to the RAFT-Stereo main forward

        return hidden_states


class MaskPredictor(raft.MaskPredictor):
    """Mask predictor to be used when upsampling the predicted depth."""
    # We add out_channels compared to raft.MaskPredictor
    def __init__(self, *, in_channels, hidden_size, out_channels, multiplier=0.25):
        super(raft.MaskPredictor, self).__init__()
        self.convrelu = Conv2dNormActivation(in_channels, hidden_size, norm_layer=None, kernel_size=3)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1, padding=0)
        self.multiplier = multiplier


class CorrBlock1d(nn.Module):
    """The row-wise correlation block.

    Creates a row-wise correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding row neighbours
    within radius
    """

    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

        self.corr_pyramid: List[Tensor] = [torch.tensor(0)]  # useless, but torchscript is otherwise confused :')

        self.out_channels = num_levels * (2 * radius + 1)

    def build_pyramid(self, fmap1, fmap2):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2) on the same row.
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)"
            )
        corr_volume = self._compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, num_channels, _ = corr_volume.shape
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, 1, w)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=(1, 2), stride=(1, 2))
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        di = di.view(1, 1, neighborhood_side_len, 1).to(centroids_coords.device)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2 but we only use the first one
        # We only consider 1d and take the first dim only
        centroids_coords = centroids_coords[:, :1].permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 1)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
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
        if corr_features.shape != expected_output_shape:
            raise ValueError(
                f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
            )

        return corr_features

    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h, w)
        fmap2 = fmap2.view(batch_size, num_channels, h, w)

        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.view(batch_size, h, w, 1, w)
        return corr / torch.sqrt(torch.tensor(num_channels))


class RaftStereo(nn.Module):
    def __init__(
        self,
        *,
        feature_encoder,
        context_encoder,
        corr_block,
        update_block,
        depth_head,
        mask_predictor=None,
        slow_fast=False,
        num_iters=12,
    ):
        """RAFT-Stereo model from
        `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.

        args:
            feature_encoder (nn.Module): The feature encoder. Its input is the concatenation of ``image1`` and ``image2``.
            context_encoder (nn.Module): The context encoder. Its input is ``image1``.
                It has multi-level output and each level will have 2 parts:

                - one part will be used as the actual "context", passed to the recurrent unit of the ``update_block``
                - one part will be used to initialize the hidden state of the of the recurrent unit of
                  the ``update_block``

            corr_block (nn.Module): The correlation block, which creates a correlation pyramid from the output of the
                ``feature_encoder``, and then indexes from this pyramid to create correlation features. It must expose
                2 methods:

                - a ``build_pyramid`` method that takes ``feature_map_1`` and ``feature_map_2`` as input (these are the
                  output of the ``feature_encoder``).
                - a ``index_pyramid`` method that takes the coordinates of the centroid pixels as input, and returns
                  the correlation features.

                It must expose an ``out_channels`` attribute.

            update_block (nn.Module): The update block, which contains the motion encoder, and the recurrent unit.
                It takes as input the hidden state of its recurrent unit, the context, the correlation
                features, and the current predicted depth. It outputs an updated hidden state
            depth_head (nn.Module): The depth head block will convert from the hidden state into changes in depth.
            mask_predictor (nn.Module, optional): Predicts the mask that will be used to upsample the predicted flow.
                If ``None`` (default), the flow is upsampled using interpolation.
        """
        super().__init__()
        _log_api_usage_once(self)

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder

        self.base_downsampling_ratio = feature_encoder.base_downsampling_ratio
        self.num_level = self.context_encoder.num_level

        self.corr_block = corr_block
        self.update_block = update_block
        self.depth_head = depth_head
        self.mask_predictor = mask_predictor

        hidden_dims = self.update_block.hidden_dims
        # Follow the original implementation to do pre convolution on the context
        # See: https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/raft_stereo.py#L32
        self.context_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dims[i], hidden_dims[i] * 3, kernel_size=3, padding=1) for i in range(self.num_level)]
        )
        self.slow_fast = slow_fast
        self.num_iters = num_iters

    def forward(self, image1, image2, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iters
        batch_size, _, h, w = image1.shape
        if (h, w) != image2.shape[-2:]:
            raise ValueError(f"input images should have the same shape, instead got ({h}, {w}) != {image2.shape[-2:]}")
        if not (h % self.base_downsampling_ratio == 0) and (w % self.base_downsampling_ratio == 0):
            raise ValueError(
                f"input image H and W should be divisible by {self.base_downsampling_ratio}, insted got {h} (h) and {w} (w)"
            )

        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        if fmap1.shape[-2:] != (h // self.base_downsampling_ratio, w // self.base_downsampling_ratio):
            raise ValueError(f"The feature encoder should downsample H and W by {self.base_downsampling_ratio}")

        self.corr_block.build_pyramid(fmap1, fmap2)

        # Multi level contexts
        context_outs = self.context_encoder(image1)

        hidden_dims = self.update_block.hidden_dims
        context_out_channels = [context_outs[i].shape[1] - hidden_dims[i] for i in range(len(context_outs))]
        hidden_states, contexts = [], []
        for i in range(len(context_outs)):
            # As in the original paper, the actual output of the context encoder is split in 2 parts:
            # - one part is used to initialize the hidden state of the recurent units of the update block
            # - the rest is the "actual" context.
            hidden_state, context = torch.split(context_outs[i], [hidden_dims[i], context_out_channels[i]], dim=1)
            hidden_states.append(torch.tanh(hidden_state))
            contexts.append(self.context_convs[i](F.relu(context)).split(split_size=[hidden_dims[i]] * 3, dim=1))

        _, Cf, Hf, Wf = fmap1.shape
        coords0 = make_coords_grid(batch_size, Hf, Wf).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, Hf, Wf).to(fmap1.device)

        depth_predictions = []
        for _ in range(num_iters):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)

            depth = coords1 - coords0
            if self.slow_fast:
                # We process lower resolution multiple times more often
                for i in range(1, self.num_level):
                    level_processed = [0] * (self.num_level - i) + [1] * i
                    hidden_states = self.update_block(
                        hidden_states, contexts, corr_features, depth, level_processed=level_processed
                    )
            hidden_states = self.update_block(
                hidden_states, contexts, corr_features, depth, level_processed=[1] * self.num_level
            )
            # Take the largest hidden_state to get the depth
            hidden_state = hidden_states[0]
            delta_depth = self.depth_head(hidden_state)
            # in stereo mode, project depth onto epipolar
            delta_depth[:, 1] = 0.0

            coords1 = coords1 + delta_depth
            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            upsampled_depth = upsample_with_mask_and_factor(
                x=(coords1 - coords0), up_mask=up_mask, factor=self.base_downsampling_ratio
            )
            depth_predictions.append(upsampled_depth[:, :1])

        return depth_predictions


def _raft_stereo(
    *,
    weights=None,
    progress=False,
    shared_encoder_weight=False,
    # Feature encoder
    feature_encoder_layers,
    feature_encoder_strides,
    feature_encoder_block,
    # Context encoder
    context_encoder_layers,
    context_encoder_strides,
    context_encoder_out_with_blocks,
    context_encoder_block,
    # Correlation block
    corr_block_num_levels,
    corr_block_radius,
    # Motion encoder
    motion_encoder_corr_layers,
    motion_encoder_flow_layers,
    motion_encoder_out_channels,
    # Update block
    update_block_hidden_dims,
    # Flow Head
    flow_head_hidden_size,
    # Mask predictor
    mask_predictor_hidden_size,
    use_mask_predictor,
    slow_fast,
    num_iters,
    **kwargs,
):

    if shared_encoder_weight:
        base_encoder = BaseEncoder(
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
    feature_encoder = FeatureEncoder(
        feature_base_encoder,
        output_dim=feature_encoder_layers[-1],
        shared_base=shared_encoder_weight,
        block=feature_encoder_block,
    )
    context_encoder = MultiLevelContextEncoder(
        context_base_encoder,
        output_dim=context_encoder_layers[-1],
        out_with_blocks=context_encoder_out_with_blocks,
        block=context_encoder_block,
    )

    feature_downsampling_ratio = feature_encoder.base_downsampling_ratio

    corr_block = CorrBlock1d(num_levels=corr_block_num_levels, radius=corr_block_radius)

    update_block = kwargs.pop("update_block", None)
    if update_block is None:
        motion_encoder = MotionEncoder(
            in_channels_corr=corr_block.out_channels,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )
        update_block = MultiLevelUpdateBlock(motion_encoder=motion_encoder, hidden_dims=update_block_hidden_dims)

    # We use the largest scale hidden_dims of update_block to get the predicted depth
    depth_head = FlowHead(
        in_channels=update_block_hidden_dims[0],
        hidden_size=flow_head_hidden_size,
    )
    mask_predictor = kwargs.pop("mask_predictor", None)
    if mask_predictor is None and use_mask_predictor:
        mask_predictor = MaskPredictor(
            in_channels=update_block.hidden_dims[0],
            hidden_size=mask_predictor_hidden_size,
            out_channels=9 * feature_downsampling_ratio * feature_downsampling_ratio,
        )

    model = RaftStereo(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        depth_head=depth_head,
        mask_predictor=mask_predictor,
        slow_fast=slow_fast,
        num_iters=num_iters,
        **kwargs,  # not really needed, all params should be consumed by now
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class Raft_Stereo_Fast_Weights(WeightsEnum):
    pass


class Raft_Stereo_Weights(WeightsEnum):
    pass


def raft_stereo_fast(*, weights: Optional[Raft_Stereo_Fast_Weights] = None, progress=True, **kwargs) -> RaftStereo:
    """RAFT-Stereo model from
    `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Fast_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Fast_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.prototype.models.depth.stereo.raft_stereo.RaftStereo``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.prototype.models.depth.stereo.Raft_Stereo_Fast_Weights
        :members:
    """

    weights = Raft_Stereo_Fast_Weights.verify(weights)

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
        context_encoder_out_with_blocks=[1, 1],
        context_encoder_block=ResidualBlock,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
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
        num_iters=7,
        **kwargs,
    )


def raft_stereo(*, weights: Optional[Raft_Stereo_Weights] = None, progress=True, **kwargs) -> RaftStereo:
    """RAFT-Stereo model from
    `RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching <https://arxiv.org/abs/2109.07547>`_.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.prototype.models.depth.stereo.Raft_Stereo_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.prototype.models.depth.stereo.raft_stereo.RaftStereo``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.prototype.models.depth.stereo.Raft_Stereo_Weights
        :members:
    """

    weights = Raft_Stereo_Weights.verify(weights)

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
        context_encoder_out_with_blocks=[1, 1, 0],
        context_encoder_block=ResidualBlock,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
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
        num_iters=32,
        **kwargs,
    )
