import functools
from typing import Any, Dict, Optional

from torch import nn
from torchvision.prototype.utils._internal import apply_recursively


class Transform(nn.Module):
    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        return apply_recursively(functools.partial(self._transform, params=params or self.get_params(sample)), sample)

    def _extra_repr_from_attrs(self, *names: str) -> str:
        return ", ".join(f"{name}={getattr(self, name)}" for name in names)
