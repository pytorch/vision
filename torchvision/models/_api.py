import importlib
import inspect
import sys
from dataclasses import dataclass, fields
from inspect import signature
from typing import Any, Callable, Dict, cast

from torchvision._utils import StrEnum

from .._internally_replaced_utils import load_state_dict_from_url


__all__ = ["WeightsEnum", "Weights", "get_weight"]


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

    def get_state_dict(self, progress: bool) -> Dict[str, Any]:
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
    Gets the weight enum value by its full name. Example: "ResNet50_Weights.IMAGENET1K_V1"

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


def _get_enum_from_fn(fn: Callable) -> WeightsEnum:
    """
    Internal method that gets the weight enum of a specific model builder method.
    Might be removed after the handle_legacy_interface is removed.

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
