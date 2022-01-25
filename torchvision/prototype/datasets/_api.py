import io
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets import home
from torchvision.prototype.datasets.decoder import raw, pil
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo, DatasetType
from torchvision.prototype.utils._internal import add_suggestion

from . import _builtin

DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    DATASETS[dataset.name] = dataset


## Should we manually register datasets instead of relying on the content of _builtings.__init__.py?
for name, obj in _builtin.__dict__.items():
    if not name.startswith("_") and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
        register(obj())


# This is exposed as 'list', but we avoid that here to not shadow the built-in 'list'
## Maybe we could call it as list_available_datasets()?
def _list() -> List[str]:
    return sorted(DATASETS.keys())


## Does this need to be public? Looks like users would only need load() and info()
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


DEFAULT_DECODER = object()

DEFAULT_DECODER_MAP: Dict[DatasetType, Callable[[io.IOBase], torch.Tensor]] = {
    DatasetType.RAW: raw,
    DatasetType.IMAGE: pil,
}


## Should DEFAULT_DECODER just be None? Or "auto"? Or "default"?
def load(
    name: str,
    *,
    decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = DEFAULT_DECODER,  # type: ignore[assignment]
    skip_integrity_check: bool = False,  ## When do we need it to be True?
    **options: Any,
) -> IterDataPipe[Dict[str, Any]]:
    dataset = find(name)

    if decoder is DEFAULT_DECODER:
        decoder = DEFAULT_DECODER_MAP.get(dataset.info.type)

    config = dataset.info.make_config(**options)
    root = os.path.join(home(), dataset.name)

    return dataset.load(root, config=config, decoder=decoder, skip_integrity_check=skip_integrity_check)
