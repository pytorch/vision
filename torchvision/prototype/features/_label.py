from typing import Dict, Any, Optional

import torch

from ._feature import Feature, _EMPTY


class Label(Feature):
    category: Optional[str]

    @classmethod
    def _prepare_meta_data(cls, data: torch.Tensor, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        if meta_data["category"] is _EMPTY:
            meta_data["category"] = None

        return meta_data
