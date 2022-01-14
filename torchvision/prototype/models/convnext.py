from typing import Any, Optional

from ...models.convnext import ConvNeXt
from ._api import WeightsEnum
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["ConvNeXt", "ConvNeXt_Weights", "convnext_tiny"]


class ConvNeXt_Weights(WeightsEnum):
    pass


@handle_legacy_interface(weights=("pretrained", None))
def convnext_tiny(*, weights: Optional[ConvNeXt_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
