from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, Dict

from ..._internally_replaced_utils import load_state_dict_from_url


__all__ = ['Weights', 'WeightEntry']


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

    def state_dict(self, progress: bool) -> Dict[str, Any]:
        return load_state_dict_from_url(self.url, progress=progress)


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
    def check_type(cls, obj: Any) -> None:
        if obj is not None and not isinstance(obj, cls) and not isinstance(obj, WeightEntry):
            raise TypeError(f"Invalid Weight class provided; expected {cls.__name__} "
                            f"but received {obj.__class__.__name__}.")

    def state_dict(self, progress: bool) -> Dict[str, Any]:
        return self.value.state_dict(progress)

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"

    def __getattr__(self, name):
        # Be able to fetch WeightEntry attributes directly
        for f in fields(WeightEntry):
            if f.name == name:
                return object.__getattribute__(self.value, name)
        return super().__getattr__(name)
