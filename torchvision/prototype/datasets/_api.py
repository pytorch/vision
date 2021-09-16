import difflib
import io
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.hub import _get_torch_home
from torch.utils.data import IterDataPipe

from torchvision.prototype.datasets.decoder import pil
from torchvision.prototype.datasets.utils import Dataset, DatasetInfo

from . import _builtin

__all__ = ["home", "register", "list", "info", "load"]


# TODO: This needs a better default
HOME = pathlib.Path(_get_torch_home()) / "datasets" / "vision"


def home(home: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    global HOME
    if home is not None:
        HOME = pathlib.Path(home).expanduser().resolve()
        return HOME

    home = os.getenv("TORCHVISION_DATASETS_HOME")
    if home is not None:
        return pathlib.Path(home)

    return HOME


DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    DATASETS[dataset.info.name] = dataset


for name, obj in _builtin.__dict__.items():
    if not name.startswith("_") and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
        register(obj())


def list() -> List[str]:
    return sorted(DATASETS.keys())


def find(name: str) -> Dataset:
    name = name.lower()
    try:
        return DATASETS[name]
    except KeyError as error:
        msg = f"Unknown dataset '{name}'."
        close_matches = difflib.get_close_matches(name, DATASETS.keys(), n=1)
        if close_matches:
            hint = f"Did you mean '{close_matches[0]}'?"
        else:
            hint = "You can use torchvision.datasets.list() to get a list of all available datasets."
        raise ValueError(f"{msg} {hint}") from error


def info(name: str) -> DatasetInfo:
    return find(name).info


def load(
    name: str,
    *,
    decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]] = pil,
    split: str = "train",
    **options: Any,
) -> IterDataPipe[Dict[str, Any]]:
    dataset = find(name)

    config = dataset.info.make_config(split=split, **options)
    root = home() / name

    return dataset.to_datapipe(root, config=config, decoder=decoder)
