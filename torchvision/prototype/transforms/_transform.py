import collections.abc
from typing import Any, cast, Dict, Mapping, Optional, Type, TypeVar

from torch import nn
from torchvision.prototype import features
from torchvision.prototype.utils._internal import FrozenMapping

DEFAULT = object()


D = TypeVar("D")
T = TypeVar("T", bound=features.Feature)


class FeatureSpecificArguments(FrozenMapping[Type[features.Feature], D]):
    def __init__(self, foo: Mapping[Type[features.Feature], D], *, default: D = None) -> None:
        super().__init__(foo, __default__=default)

    def get(  # type: ignore[override]
        self,
        key: Type[features.Feature],
        default: D = DEFAULT,  # type: ignore[assignment]
    ) -> D:
        return super().get(key, default if default is not DEFAULT else self["__default__"])  # type: ignore[index]


class Transform(nn.Module):
    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _dispatch(self, feature: T, **params: Any) -> T:
        raise NotImplementedError

    def _apply_recursively(self, sample: Any, *, params: Dict[str, Any]) -> Any:
        # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
        # "a" == "a"[0][0]...
        if isinstance(sample, collections.abc.Sequence) and not isinstance(sample, str):
            return [self._apply_recursively(item, params=params) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: self._apply_recursively(item, params=params) for name, item in sample.items()}
        else:
            feature_type = type(sample)
            if not (issubclass(feature_type, features.Feature) and feature_type is not features.Feature):
                return sample

            return self._dispatch(
                cast(features.Feature, sample),
                **{
                    key: value.get(feature_type) if isinstance(value, FeatureSpecificArguments) else value
                    for key, value in params.items()
                },
            )

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self._last_params = self.get_params(sample)
        return self._apply_recursively(sample, params=params)
