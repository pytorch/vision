from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict

from ..._internally_replaced_utils import load_state_dict_from_url


__all__ = ["Weights", "WeightEntry", "get_weight"]


@dataclass
class WeightEntry:
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


class Weights(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `WeightEntry`.

    Args:
        value (WeightEntry): The data class entry with the weight information.
    """

    def __init__(self, value: WeightEntry):
        self._value_ = value

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls.from_str(obj)
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    @classmethod
    def from_str(cls, value: str) -> "Weights":
        for v in cls:
            if v._name_ == value:
                return v
        raise ValueError(f"Invalid value {value} for enum {cls.__name__}.")

    def get_state_dict(self, progress: bool) -> OrderedDict:
        return load_state_dict_from_url(self.url, progress=progress)

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"

    def __getattr__(self, name):
        # Be able to fetch WeightEntry attributes directly
        for f in fields(WeightEntry):
            if f.name == name:
                return object.__getattribute__(self.value, name)
        return super().__getattr__(name)


def get_weight(fn: Callable, weight_name: str) -> Weights:
    """
    Gets the weight enum of a specific model builder method and weight name combination.

    Args:
        fn (Callable): The builder method used to create the model.
        weight_name (str): The name of the weight enum entry of the specific model.

    Returns:
        Weights: The requested weight enum.
    """
    sig = signature(fn)
    if "weights" not in sig.parameters:
        raise ValueError("The method is missing the 'weights' parameter.")

    ann = signature(fn).parameters["weights"].annotation
    weights_class = None
    if isinstance(ann, type) and issubclass(ann, Weights):
        weights_class = ann
    else:
        # handle cases like Union[Optional, T]
        # TODO: Replace ann.__args__ with typing.get_args(ann) after python >= 3.8
        for t in ann.__args__:  # type: ignore[union-attr]
            if isinstance(t, type) and issubclass(t, Weights):
                # ensure the name exists. handles builders with multiple types of weights like in quantization
                try:
                    t.from_str(weight_name)
                except ValueError:
                    continue
                weights_class = t
                break

    if weights_class is None:
        raise ValueError(
            "The weight class for the specific method couldn't be retrieved. Make sure the typing info is correct."
        )

    return weights_class.from_str(weight_name)
