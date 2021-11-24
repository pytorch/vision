import warnings
from typing import Any, Optional

from ._api import Weights
from ...models.vision_transformer import VisionTransformer


__all__ = [
    "VisionTransformer",
    "VisionTransformer_B_16Weights",
    "VisionTransformer_B_32Weights",
    "VisionTransformer_L_16Weights",
    "VisionTransformer_L_32Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
]



class VisionTransformer_B_16Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in vit_b_16
    pass


class VisionTransformer_B_32Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in vit_b_32
    pass


class VisionTransformer_L_16Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in vit_l_16
    pass


class VisionTransformer_L_32Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in vit_l_32
    pass


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[Weights],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def vit_b_16(
    weights: Optional[VisionTransformer_B_16Weights] = None, progress: bool = True, **kwargs: Any
) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (VisionTransformer_B_16Weights, optional): If not None, returns a model pre-trained on ImageNet. Default: None.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True.
    """
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type vit_b_16")

    weights = VisionTransformer_B_16Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def vit_b_32(
    weights: Optional[VisionTransformer_B_32Weights] = None, progress: bool = True, **kwargs: Any
) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (VisionTransformer_B_32Weights, optional): If not None, returns a model pre-trained on ImageNet. Default: None.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True.
    """
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type vit_b_32")

    weights = VisionTransformer_B_32Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def vit_l_16(
    weights: Optional[VisionTransformer_L_16Weights] = None, progress: bool = True, **kwargs: Any
) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (VisionTransformer_L_16Weights, optional): If not None, returns a model pre-trained on ImageNet. Default: None.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True.
    """
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type vit_l_16")

    weights = VisionTransformer_L_16Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def vit_l_32(
    weights: Optional[VisionTransformer_B_32Weights] = None, progress: bool = True, **kwargs: Any
) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (VisionTransformer_L_16Weights, optional): If not None, returns a model pre-trained on ImageNet. Default: None.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True.
    """
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type vit_l_32")

    weights = VisionTransformer_L_32Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )
