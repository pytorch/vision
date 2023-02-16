from typing import Any, Dict

import torch

from torch.nn.functional import one_hot

from torchvision.prototype import datapoints as proto_datapoints
from torchvision.transforms.v2 import Transform


class LabelToOneHot(Transform):
    _transformed_types = (proto_datapoints.Label,)

    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def _transform(self, inpt: proto_datapoints.Label, params: Dict[str, Any]) -> proto_datapoints.OneHotLabel:
        num_categories = self.num_categories
        if num_categories == -1 and inpt.categories is not None:
            num_categories = len(inpt.categories)
        output = one_hot(inpt.as_subclass(torch.Tensor), num_classes=num_categories)
        return proto_datapoints.OneHotLabel(output, categories=inpt.categories)

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"
