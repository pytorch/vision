from typing import Any, Dict, Optional

from torch import nn
from torchvision.prototype.utils._internal import apply_recursively

from .functional._utils import Dispatcher


class Transform(nn.Module):
    _DISPATCHER: Optional[Dispatcher] = None

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _supports(self, obj: Any) -> bool:
        if not self._DISPATCHER:
            raise NotImplementedError()

        return obj in self._DISPATCHER

    def _dispatch(self, input: Any, params: Dict[str, Any]) -> Any:
        if not self._DISPATCHER:
            raise NotImplementedError()

        return self._DISPATCHER(input, **params)

    def _transform_recursively(self, sample: Any, params: Dict[str, Any]) -> Any:
        def transform(input: Any) -> Any:
            if not self._supports(input):
                return sample

            return self._dispatch(input, params)

        return apply_recursively(transform, sample)

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        return self._transform_recursively(sample, params or self.get_params(sample))

    def _extra_repr_from_attrs(self, *names: str) -> str:
        return ", ".join(f"{name}={getattr(self, name)}" for name in names)


class ConstantParamTransform(Transform):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self._params = params

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return self._params

    def extra_repr(self) -> str:
        return ", ".join(f"{param}={value}" for param, value in sorted(self._params.items()))
