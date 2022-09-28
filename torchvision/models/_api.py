import importlib
import inspect
import sys
from dataclasses import dataclass, fields
from inspect import signature
from types import ModuleType
from typing import Any, Callable, cast, Dict, List, Mapping, Optional, TypeVar, Union

from torch import nn

from torchvision._utils import StrEnum

from .._internally_replaced_utils import load_state_dict_from_url


__all__ = ["WeightsEnum", "Weights", "get_model", "get_model_builder", "get_model_weights", "get_weight", "list_models"]


@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    transforms: Callable
    meta: Dict[str, Any]


class WeightsEnum(StrEnum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    def __init__(self, value: Weights):
        self._value_ = value

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls.from_str(obj.replace(cls.__name__ + ".", ""))
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, progress: bool) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, progress=progress)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    def __getattr__(self, name):
        # Be able to fetch Weights attributes directly
        for f in fields(Weights):
            if f.name == name:
                return object.__getattribute__(self.value, name)
        return super().__getattr__(name)


def get_weight(name: str) -> WeightsEnum:
    """
    Gets the weights enum value by its full name. Example: "ResNet50_Weights.IMAGENET1K_V1"

    .. betastatus:: function

    Args:
        name (str): The name of the weight enum entry.

    Returns:
        WeightsEnum: The requested weight enum.
    """
    try:
        enum_name, value_name = name.split(".")
    except ValueError:
        raise ValueError(f"Invalid weight name provided: '{name}'.")

    base_module_name = ".".join(sys.modules[__name__].__name__.split(".")[:-1])
    base_module = importlib.import_module(base_module_name)
    model_modules = [base_module] + [
        x[1] for x in inspect.getmembers(base_module, inspect.ismodule) if x[1].__file__.endswith("__init__.py")
    ]

    weights_enum = None
    for m in model_modules:
        potential_class = m.__dict__.get(enum_name, None)
        if potential_class is not None and issubclass(potential_class, WeightsEnum):
            weights_enum = potential_class
            break

    if weights_enum is None:
        raise ValueError(f"The weight enum '{enum_name}' for the specific method couldn't be retrieved.")

    return weights_enum.from_str(value_name)


def get_model_weights(name: Union[Callable, str]) -> WeightsEnum:
    """
    Retuns the weights enum class associated to the given model.

    .. betastatus:: function

    Args:
        name (callable or str): The model builder function or the name under which it is registered.

    Returns:
        weights_enum (WeightsEnum): The weights enum class associated with the model.
    """
    model = get_model_builder(name) if isinstance(name, str) else name
    return _get_enum_from_fn(model)


def _get_enum_from_fn(fn: Callable) -> WeightsEnum:
    """
    Internal method that gets the weight enum of a specific model builder method.

    Args:
        fn (Callable): The builder method used to create the model.
        weight_name (str): The name of the weight enum entry of the specific model.
    Returns:
        WeightsEnum: The requested weight enum.
    """
    sig = signature(fn)
    if "weights" not in sig.parameters:
        raise ValueError("The method is missing the 'weights' argument.")

    ann = signature(fn).parameters["weights"].annotation
    weights_enum = None
    if isinstance(ann, type) and issubclass(ann, WeightsEnum):
        weights_enum = ann
    else:
        # handle cases like Union[Optional, T]
        # TODO: Replace ann.__args__ with typing.get_args(ann) after python >= 3.8
        for t in ann.__args__:  # type: ignore[union-attr]
            if isinstance(t, type) and issubclass(t, WeightsEnum):
                weights_enum = t
                break

    if weights_enum is None:
        raise ValueError(
            "The WeightsEnum class for the specific method couldn't be retrieved. Make sure the typing info is correct."
        )

    return cast(WeightsEnum, weights_enum)


M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}


def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


def list_models(module: Optional[ModuleType] = None) -> List[str]:
    """
    Returns a list with the names of registered models.

    .. betastatus:: function

    Args:
        module (ModuleType, optional): The module from which we want to extract the available models.

    Returns:
        models (list): A list with the names of available models.
    """
    models = [
        k for k, v in BUILTIN_MODELS.items() if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    ]
    return sorted(models)


def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    .. betastatus:: function

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def get_model(name: str, **config: Any) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    .. betastatus:: function

    Args:
        name (str): The name under which the model is registered.
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_model_builder(name)
    return fn(**config)
