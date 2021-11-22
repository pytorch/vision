from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F

from ..._internally_replaced_utils import load_state_dict_from_url
from ...utils import _log_api_usage_once


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str], progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)
