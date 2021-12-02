import functools
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union

from torch import nn
from torchvision.prototype.utils._internal import kwonly_to_pos_or_kw

from ._api import WeightsEnum

W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Any, Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Any, Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with the value of ``pretrained`` and the other keyword arguments. For example:
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument already use
                # the new style (not a boolean like pretrained did), there is nothing to do.
                if (weights_param not in kwargs and pretrained_param not in kwargs) or not isinstance(
                    kwargs.get(weights_param, False), bool
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw use the current signature to
                # infer the names of positionally passed arguments and thus has no knowledge that there used to be a
                # pretrained parameter.
                pretrained_positional = weights_param in kwargs
                pretrained_arg = kwargs.pop(weights_param if pretrained_positional else pretrained_param)
                if pretrained_arg:
                    default_weights_arg = default(pretrained_arg, kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated, please use '{weights_param}' instead."
                    )

                msg = (
                    f"Boolean arguments for '{weights_param}' are deprecated. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.default` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


# TODO: remove this in favor of handle_legacy_interface
def _deprecated_param(
    kwargs: Dict[str, Any], deprecated_param: str, new_param: str, default_value: Optional[W]
) -> Optional[W]:
    warnings.warn(f"The parameter '{deprecated_param}' is deprecated, please use '{new_param}' instead.")
    if kwargs.pop(deprecated_param):
        if default_value is not None:
            return default_value
        else:
            raise ValueError("No checkpoint is available for model.")
    else:
        return None


# TODO: remove this in favor of handle_legacy_interface
def _deprecated_positional(kwargs: Dict[str, Any], deprecated_param: str, new_param: str, default_value: V) -> None:
    warnings.warn(
        f"The positional parameter '{deprecated_param}' is deprecated, please use keyword parameter '{new_param}'"
        + " instead."
    )
    kwargs[deprecated_param] = default_value


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: Optional[V], new_value: V) -> V:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {param} instead.")
    return new_value
