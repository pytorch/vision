from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchvision.prototype import datapoints
from torchvision.prototype.transforms.utils import check_type
from torchvision.utils import _log_api_usage_once


class Transform(nn.Module):

    # Class attribute defining transformed types. Other types are passed-through without any transformation
    # We support both Types and callables that are able to do further checks on the type of the input.
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor, PIL.Image.Image)

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        pass

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self._check_inputs(flat_inputs)

        params = self._get_params(flat_inputs)

        flat_outputs = [
            self._transform(inpt, params) if check_type(inpt, self._transformed_types) else inpt for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)

    # FIXME explain everything below

    _v1_transform_cls: Optional[Type[nn.Module]] = None

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        common_attrs = nn.Module().__dict__.keys()
        params = {
            attr: value
            for attr, value in self.__dict__.items()
            if not attr.startswith("_") and attr not in common_attrs
        }

        if "fill" not in params:
            return params

        fill_type_dict = params.pop("fill")
        fill_candidates = [fill_type_dict[typ] for typ in [torch.Tensor, datapoints.Image]]
        try:
            fill = next(filter(lambda candidate: candidate is not None, fill_candidates))
        except StopIteration:
            fill = None
        params["fill"] = fill
        return params

    def __init_subclass__(cls) -> None:
        if cls._v1_transform_cls is not None and hasattr(cls._v1_transform_cls, "get_params"):
            cls.get_params = cls._v1_transform_cls.get_params  # type: ignore[attr-defined]

    def __prepare_scriptable__(self) -> nn.Module:
        if self._v1_transform_cls is None:
            # FIXME: add more information
            raise RuntimeError("This transform cannot be scripted!")

        return self._v1_transform_cls(**self._extract_params_for_v1_transform())


class _RandomApplyTransform(Transform):
    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")

        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        if torch.rand(1) >= self.p:
            return inputs

        params = self._get_params(flat_inputs)

        flat_outputs = [
            self._transform(inpt, params) if check_type(inpt, self._transformed_types) else inpt for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)
