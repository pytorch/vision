import functools
from typing import Any, cast, Dict, Mapping, Optional, Type, TypeVar, Set

import torch
from torch import nn
from torchvision.prototype import features
from torchvision.prototype.utils._internal import FrozenMapping, apply_recursively

from .functional.utils import dispatch

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
    _DISPATCH: Optional[dispatch] = None
    _NATIVE_FEATURE_TYPES = {
        features.Image,
        features.BoundingBox,
        features.EncodedImage,
        features.Label,
        features.OneHotLabel,
    }
    _NO_OP_FEATURE_TYPES: Set[Type[features.Feature]] = set()

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def supports(self, obj: Any) -> bool:
        if not self._DISPATCH:
            raise NotImplementedError()

        return self._DISPATCH.supports(obj)

    def register_no_op_feature_type(self, feature_type: Type[features.Feature]) -> None:
        self._NO_OP_FEATURE_TYPES.add(feature_type)

    def _dispatch(self, feature: T, params: Dict[str, Any]) -> Any:
        if not self._DISPATCH:
            raise NotImplementedError()

        return self._DISPATCH(feature, **params)

    def _transform(self, sample: Any, *, params: Dict[str, Any]) -> Any:
        feature_type = type(sample)
        if not (issubclass(feature_type, features.Feature) and feature_type is not features.Feature):
            return sample

        if not self.supports(feature_type):
            if feature_type not in (self._NATIVE_FEATURE_TYPES | self._NO_OP_FEATURE_TYPES):
                # This prevents subtle bugs that would turn this transform into a no-op for foreign features
                raise TypeError(
                    f"{type(self).__name__}() does not support feature inputs of type {feature_type.__name__}. "
                    f"If you want {type(self).__name__}() to return inputs of this type unchanged, "
                    f"invoke {type(self).__name__}().register_no_op_feature_type({feature_type.__name__}) "
                    f"once before you start using it."
                )

            return sample

        return self._dispatch(
            cast(features.Feature, sample),
            {
                key: value.get(feature_type) if isinstance(value, FeatureSpecificArguments) else value
                for key, value in params.items()
            },
        )

    def _transform_recursively(self, sample: Any, params: Dict[str, Any]) -> Any:
        return apply_recursively(functools.partial(self._transform, params=params), sample)

    def _tensor_to_image(self, sample: Any) -> Any:
        if not isinstance(sample, torch.Tensor) or isinstance(sample, features.Feature):
            return sample

        return features.Image(sample)

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        # This is for BC
        sample = apply_recursively(self._tensor_to_image, sample)
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
