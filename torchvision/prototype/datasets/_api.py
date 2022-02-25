import pathlib
from typing import Any, Dict, List, Callable, Type, Optional, Union, TypeVar

from torchvision.prototype.datasets import home
from torchvision.prototype.datasets.utils import Dataset2
from torchvision.prototype.utils._internal import add_suggestion


T = TypeVar("T")
D = TypeVar("D", bound=Type[Dataset2])

BUILTIN_INFOS: Dict[str, Dict[str, Any]] = {}


BUILTIN_DATASETS = {}


def register_dataset(name: str, *, info: Optional[Dict[str, Any]] = None) -> Callable[[D], D]:
    if info is None:
        info = dict()

    def wrapper(dataset_cls: D) -> D:
        BUILTIN_INFOS[name] = info  # type: ignore[assignment]
        BUILTIN_DATASETS[name] = dataset_cls

        dataset_cls._NAME = name
        dataset_cls._INFO = info  # type: ignore[assignment]

        return dataset_cls

    return wrapper


def list_datasets() -> List[str]:
    return sorted(BUILTIN_DATASETS.keys())


def find(dct: Dict[str, T], name: str) -> T:
    name = name.lower()
    try:
        return dct[name]
    except KeyError as error:
        raise ValueError(
            add_suggestion(
                f"Unknown dataset '{name}'.",
                word=name,
                possibilities=dct.keys(),
                alternative_hint=lambda _: (
                    "You can use torchvision.datasets.list_datasets() to get a list of all available datasets."
                ),
            )
        ) from error


def info(name: str) -> Dict[str, Any]:
    return find(BUILTIN_INFOS, name)


def load(name: str, *, root: Optional[Union[str, pathlib.Path]] = None, **config: Any) -> Dataset2:
    dataset_cls = find(BUILTIN_DATASETS, name)

    if root is None:
        root = pathlib.Path(home()) / name

    return dataset_cls(root, **config)
