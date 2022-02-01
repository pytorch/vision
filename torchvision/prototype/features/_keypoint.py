from typing import Dict, Any, Tuple, Optional, Sequence

import torch

from ._feature import Feature, DEFAULT


class KeyPoint(Feature):
    names: Optional[Sequence[str]]

    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if tensor.shape[-1] != 2:
            raise ValueError
        return tensor

    @classmethod
    def _parse_meta_data(
        cls,
        names: Optional[Sequence[str]] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        return dict(names=(names, None))
