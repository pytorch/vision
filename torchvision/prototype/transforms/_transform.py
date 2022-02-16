import enum
import functools
from typing import Any, Dict, Optional

from torch import nn
from torchvision.prototype.utils._internal import apply_recursively
from torchvision.utils import _log_api_usage_once

from .functional._utils import Dispatcher


class Transform(nn.Module):
    _DISPATCHER: Optional[Dispatcher] = None

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if not self._DISPATCHER:
            raise NotImplementedError()

        if input not in self._DISPATCHER:
            return input

        return self._DISPATCHER(input, **params)

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        return apply_recursively(functools.partial(self._transform, params=params or self._get_params(sample)), sample)

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)
