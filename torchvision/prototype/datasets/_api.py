import io
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets import home
from torchvision.prototype.datasets.decoder import pil
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetType
from torchvision.prototype.datasets.utils._internal import add_suggestion

from . import _builtin

DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    DATASETS[dataset.name] = dataset


for name, obj in _builtin.__dict__.items():
    if not name.startswith("_") and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
        register(obj())


# This is exposed as 'list', but we avoid that here to not shadow the built-in 'list'
def _list() -> List[str]:
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
                    "You can use torchvision.datasets.list() to get a list of all available datasets."
                ),
            )
        ) from error


def info(name: str) -> DatasetInfo:
    return find(name).info


default = object()

DEFAULT_DECODER: Dict[DatasetType, Callable[[io.IOBase], torch.Tensor]] = {
    DatasetType.IMAGE: pil,
}


def load(
    name: str,
    *,
    decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = default,  # type: ignore[assignment]
    split: str = "train",
    **options: Any,
) -> IterDataPipe[Dict[str, Any]]:
    dataset = find(name)

    if decoder is default:
        decoder = DEFAULT_DECODER.get(dataset.info.type)

    config = dataset.info.make_config(split=split, **options)
    root = home() / name

    return dataset.to_datapipe(root, config=config, decoder=decoder)
