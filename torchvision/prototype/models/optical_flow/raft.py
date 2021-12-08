from typing import Optional

from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.models.optical_flow import RAFT
from torchvision.models.optical_flow.raft import _raft, BottleneckBlock, ResidualBlock
from torchvision.prototype.transforms import RaftEval

from .._api import WeightsEnum
from .._api import Weights
from .._utils import handle_legacy_interface


__all__ = (
    "RAFT",
    "raft_large",
    "raft_small",
    "Raft_Large_Weights",
    "Raft_Small_Weights",
)


class Raft_Large_Weights(WeightsEnum):
    C_T_V2 = Weights(
        # Chairs + Things
        url="https://download.pytorch.org/models/raft_large_C_T_V2-1bb1363a.pth",
        transforms=RaftEval,
        meta={
            "recipe": "",  # TODO
            "sintel_train_cleanpass_epe": 1.3822,
            "sintel_train_finalpass_epe": 2.7161,
        },
    )

    # C_T_SKHT_V1 = Weights(
    #     # Chairs + Things + Sintel fine-tuning, i.e.:
    #     # Chairs + Things + (Sintel + Kitti + HD1K + Things_clean)
    #     # Corresponds to the C+T+S+K+H on paper with fine-tuning on Sintel
    #     url="",
    #     transforms=RaftEval,
    #     meta={
    #         "recipe": "",
    #         "epe": -1234,
    #     },
    # )

    # C_T_SKHT_K_V1 = Weights(
    #     # Chairs + Things + Sintel fine-tuning + Kitti fine-tuning i.e.:
    #     # Chairs + Things + (Sintel + Kitti + HD1K + Things_clean) + Kitti
    #     # Same as CT_SKHT with extra fine-tuning on Kitti
    #     # Corresponds to the C+T+S+K+H on paper with fine-tuning on Sintel and then on Kitti
    #     url="",
    #     transforms=RaftEval,
    #     meta={
    #         "recipe": "",
    #         "epe": -1234,
    #     },
    # )

    default = C_T_V2


class Raft_Small_Weights(WeightsEnum):
    pass
    # C_T_V1 = Weights(
    #     url="",  # TODO
    #     transforms=RaftEval,
    #     meta={
    #         "recipe": "",
    #         "epe": -1234,
    #     },
    # )
    # default = C_T_V1


def _raft_builder(
    *,
    weights,
    progress,
    # Feature encoder
    feature_encoder_layers,
    feature_encoder_block,
    feature_encoder_norm_layer,
    # Context encoder
    context_encoder_layers,
    context_encoder_block,
    context_encoder_norm_layer,
    # Correlation block
    corr_block_num_levels,
    corr_block_radius,
    # Motion encoder
    motion_encoder_corr_layers,
    motion_encoder_flow_layers,
    motion_encoder_out_channels,
    # Recurrent block
    recurrent_block_hidden_state_size,
    recurrent_block_kernel_size,
    recurrent_block_padding,
    # Flow Head
    flow_head_hidden_size,
    # Mask predictor
    use_mask_predictor,
    **kwargs,
):
    model = _raft(
        # Feature encoder
        feature_encoder_layers=feature_encoder_layers,
        feature_encoder_block=feature_encoder_block,
        feature_encoder_norm_layer=feature_encoder_norm_layer,
        # Context encoder
        context_encoder_layers=context_encoder_layers,
        context_encoder_block=context_encoder_block,
        context_encoder_norm_layer=context_encoder_norm_layer,
        # Correlation block
        corr_block_num_levels=corr_block_num_levels,
        corr_block_radius=corr_block_radius,
        # Motion encoder
        motion_encoder_corr_layers=motion_encoder_corr_layers,
        motion_encoder_flow_layers=motion_encoder_flow_layers,
        motion_encoder_out_channels=motion_encoder_out_channels,
        # Recurrent block
        recurrent_block_hidden_state_size=recurrent_block_hidden_state_size,
        recurrent_block_kernel_size=recurrent_block_kernel_size,
        recurrent_block_padding=recurrent_block_padding,
        # Flow head
        flow_head_hidden_size=flow_head_hidden_size,
        # Mask predictor
        use_mask_predictor=use_mask_predictor,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", Raft_Large_Weights.C_T_V2))
def raft_large(*, weights: Optional[Raft_Large_Weights] = None, progress=True, **kwargs):
    """RAFT model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`_.

    Args:
        weights(Raft_Large_weights, optional): pretrained weights to use.
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs (dict): Parameters that will be passed to the :class:`~torchvision.models.optical_flow.RAFT` class
            to override any default.

    Returns:
        nn.Module: The model.
    """

    weights = Raft_Large_Weights.verify(weights)

    return _raft_builder(
        weights=weights,
        progress=progress,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_block=ResidualBlock,
        feature_encoder_norm_layer=InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_block=ResidualBlock,
        context_encoder_norm_layer=BatchNorm2d,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(256, 192),
        motion_encoder_flow_layers=(128, 64),
        motion_encoder_out_channels=128,
        # Recurrent block
        recurrent_block_hidden_state_size=128,
        recurrent_block_kernel_size=((1, 5), (5, 1)),
        recurrent_block_padding=((0, 2), (2, 0)),
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        use_mask_predictor=True,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", None))
def raft_small(*, weights: Optional[Raft_Small_Weights] = None, progress=True, **kwargs):
    """RAFT "small" model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`_.

    Args:
        weights(Raft_Small_weights, optinal): TODO not implemented yet
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs (dict): Parameters that will be passed to the :class:`~torchvision.models.optical_flow.RAFT` class
            to override any default.

    Returns:
        nn.Module: The model.

    """

    weights = Raft_Small_Weights.verify(weights)

    return _raft_builder(
        weights=weights,
        progress=progress,
        # Feature encoder
        feature_encoder_layers=(32, 32, 64, 96, 128),
        feature_encoder_block=BottleneckBlock,
        feature_encoder_norm_layer=InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(32, 32, 64, 96, 160),
        context_encoder_block=BottleneckBlock,
        context_encoder_norm_layer=None,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=3,
        # Motion encoder
        motion_encoder_corr_layers=(96,),
        motion_encoder_flow_layers=(64, 32),
        motion_encoder_out_channels=82,
        # Recurrent block
        recurrent_block_hidden_state_size=96,
        recurrent_block_kernel_size=(3,),
        recurrent_block_padding=(1,),
        # Flow head
        flow_head_hidden_size=128,
        # Mask predictor
        use_mask_predictor=False,
        **kwargs,
    )
