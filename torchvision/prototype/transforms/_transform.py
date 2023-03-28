from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
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

    # This attribute should be set on all transforms that have a v1 equivalent. Doing so enables two things:
    # 1. In case the v1 transform has a static `get_params` method, it will also be available under the same name on
    #    the v2 transform. See `__init_subclass__` for details.
    # 2. The v2 transform will be JIT scriptable. See `_extract_params_for_v1_transform` and `__prepare_scriptable__`
    #    for details.
    _v1_transform_cls: Optional[Type[nn.Module]] = None

    def __init_subclass__(cls) -> None:
        # Since `get_params` is a `@staticmethod`, we have to bind it to the class itself rather than to an instance.
        # This method is called after subclassing has happened, i.e. `cls` is the subclass, e.g. `Resize`.
        if cls._v1_transform_cls is not None and hasattr(cls._v1_transform_cls, "get_params"):
            cls.get_params = staticmethod(cls._v1_transform_cls.get_params)  # type: ignore[attr-defined]

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        # This method is called by `__prepare_scriptable__` to instantiate the equivalent v1 transform from the current
        # v2 transform instance. It does two things:
        # 1. Extract all available public attributes that are specific to that transform and not `nn.Module` in general
        # 2. If available handle the `fill` attribute for v1 compatibility (see below for details)
        # Overwrite this method on the v2 transform class if the above is not sufficient. For example, this might happen
        # if the v2 transform introduced new parameters that are not support by the v1 transform.
        common_attrs = nn.Module().__dict__.keys()
        params = {
            attr: value
            for attr, value in self.__dict__.items()
            if not attr.startswith("_") and attr not in common_attrs
        }

        # transforms v2 has a more complex handling for the `fill` parameter than v1. By default, the input is parsed
        # with `prototype.transforms._utils._setup_fill_arg()`, which returns a defaultdict that holds the fill value
        # for the different datapoint types. Below we extract the value for tensors and return that together with the
        # other params.
        # This is needed for `Pad`, `ElasticTransform`, `RandomAffine`, `RandomCrop`, `RandomPerspective` and
        # `RandomRotation`
        if "fill" in params:
            fill_type_defaultdict = params.pop("fill")
            params["fill"] = fill_type_defaultdict[torch.Tensor]

        return params

    def __prepare_scriptable__(self) -> nn.Module:
        # This method is called early on when `torch.jit.script`'ing an `nn.Module` instance. If it succeeds, the return
        # value is used for scripting over the original object that should have been scripted. Since the v1 transforms
        # are JIT scriptable, and we made sure that for single image inputs v1 and v2 are equivalent, we just return the
        # equivalent v1 transform here. This of course only makes transforms v2 JIT scriptable as long as transforms v1
        # is around.
        if self._v1_transform_cls is None:
            raise RuntimeError(
                f"Transform {type(self.__name__)} cannot be JIT scripted. "
                f"This is only support for backward compatibility with transforms which already in v1."
                f"For torchscript support (on tensors only), you can use the functional API instead."
            )

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
