import os
import pathlib
from typing import Any, Dict, List, Callable, Type, Optional, Union

from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets import home
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, Dataset2
from torchvision.prototype.utils._internal import add_suggestion


DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    DATASETS[dataset.name] = dataset


def list_datasets() -> List[str]:
    return sorted(DATASETS.keys())


def find(name: str) -> Dataset:
    name = name.lower()
    try:
        return DATASETS[name]
    except KeyError as error:
        raise ValueError(
            add_suggestion(
                f"Unknown dataset '{name}'.",
                word=name,
                possibilities=DATASETS.keys(),
                alternative_hint=lambda _: (
                    "You can use torchvision.datasets.list_datasets() to get a list of all available datasets."
                ),
            )
        ) from error


def info(name: str) -> DatasetInfo:
    return find(name).info


def load(
    name: str,
    *,
    skip_integrity_check: bool = False,
    **options: Any,
) -> IterDataPipe[Dict[str, Any]]:
    dataset = find(name)

    config = dataset.info.make_config(**options)
    root = os.path.join(home(), dataset.name)

    return dataset.load(root, config=config, skip_integrity_check=skip_integrity_check)


BUILTIN_INFOS: Dict[str, Dict[str, Any]] = {}


def register_info(name: str) -> Callable[[Callable[[], Dict[str, Any]]], Callable[[], Dict[str, Any]]]:
    def wrapper(fn: Callable[[], Dict[str, Any]]) -> Callable[[], Dict[str, Any]]:
        BUILTIN_INFOS[name] = fn()
        return fn

    return wrapper


def info2(name: str) -> Dict[str, Any]:
    try:
        return BUILTIN_INFOS[name]
    except KeyError:
        raise ValueError


BUILTIN_DATASETS = {}


def register_dataset(name: str) -> Callable[[Type], Type]:
    def wrapper(dataset_cls: Type) -> Type:
        if not issubclass(dataset_cls, Dataset2):
            raise TypeError

        BUILTIN_DATASETS[name] = dataset_cls

        return dataset_cls

    return wrapper


def load2(name: str, *, root: Optional[Union[str, pathlib.Path]] = None, **options: Any) -> Dataset2:
    try:
        dataset_cls = BUILTIN_DATASETS[name]
    except KeyError:
        raise ValueError

    if root is None:
        root = pathlib.Path(home()) / name

    return dataset_cls(root, **options)


from . import _builtin

for name, obj in _builtin.__dict__.items():
    if not name.startswith("_") and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
        register(obj())
