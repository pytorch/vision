import os
from typing import Any, Dict, List

from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets import home
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo
from torchvision.prototype.utils._internal import add_suggestion

from . import _builtin

DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    DATASETS[dataset.name] = dataset


for name, obj in _builtin.__dict__.items():
    if not name.startswith("_") and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
        register(obj())


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
