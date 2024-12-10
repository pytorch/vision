from typing import Any, Dict

import torch

from torch.nn.functional import one_hot

from torchvision.prototype import tv_tensors as proto_tv_tensors
from torchvision.transforms.v2 import Transform


class LabelToOneHot(Transform):
    _transformed_types = (proto_tv_tensors.Label,)

    def __init__(self, num_categories: int = -1):
        super().__init__()
        self.num_categories = num_categories

    def transform(self, inpt: proto_tv_tensors.Label, params: Dict[str, Any]) -> proto_tv_tensors.OneHotLabel:
        num_categories = self.num_categories
        if num_categories == -1 and inpt.categories is not None:
            num_categories = len(inpt.categories)
        output = one_hot(inpt.as_subclass(torch.Tensor), num_classes=num_categories)
        return proto_tv_tensors.OneHotLabel(output, categories=inpt.categories)

    def extra_repr(self) -> str:
        if self.num_categories == -1:
            return ""

        return f"num_categories={self.num_categories}"
