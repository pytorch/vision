from typing import Any, Optional

from ...models.convnext import ConvNeXt, CNBlockConfig
from ._api import WeightsEnum
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["ConvNeXt", "ConvNeXt_Tiny_Weights", "convnext_tiny"]


class ConvNeXt_Tiny_Weights(WeightsEnum):
    pass


@handle_legacy_interface(weights=("pretrained", None))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    model = ConvNeXt(block_setting, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
