import abc
import csv
import enum
import importlib
import io
import os
import pathlib
from typing import Callable, Any, Dict, List, Optional, Sequence, Union, Collection

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.utils._internal import FrozenBunch, add_suggestion, sequence_to_str

from .._home import use_sharded_dataset
from ._internal import BUILTIN_DIR, _make_sharded_datapipe
from ._resource import OnlineResource


class DatasetType(enum.Enum):
    RAW = enum.auto()
    IMAGE = enum.auto()


# TODO: this is just a dummy to not change the imports everywhere before the design is finalized
class DatasetInfo:
    pass


class DatasetConfig(FrozenBunch):
    # This needs to be Frozen because we often pass configs as partial(func, config=config)
    # and partial() requires the parameters to be hashable.
    pass


SENTINEL = object()


class DatasetOption:
    def __init__(self, name: str, valid: Sequence, *, default: Any = SENTINEL, doc: str = "{options}") -> None:
        self.name = name
        if not valid:
            raise ValueError
        self.valid = valid
        self.default = valid[0] if default is SENTINEL else default
        if "{options}" in doc:
            options = sequence_to_str(
                [str(arg) + (" (default)" if arg == self.default else "") for arg in valid],
                separate_last="or ",
            )
            doc = doc.format(options=f"Can be one of {options}.")
        self.doc = doc


class Dataset(abc.ABC):
    def __init__(
        self,
        name: str,
        *options: DatasetOption,
        type: Union[str, DatasetType],
        dependencies: Collection[str] = (),
        categories: Optional[Union[int, Sequence[str], str, pathlib.Path]] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
    ) -> None:

        self.name = name.lower()
        self.type = DatasetType[type.upper()] if isinstance(type, str) else type

        self.options = options
        self._options_map = {option.name: option for option in options}
        self.default_config = DatasetConfig({option.name: option.default for option in options})

        self.dependecies = dependencies

        if categories is None:
            path = BUILTIN_DIR / f"{self.name}.categories"
            categories = path if path.exists() else []
        if isinstance(categories, int):
            categories = [str(label) for label in range(categories)]
        elif isinstance(categories, (str, pathlib.Path)):
            path = pathlib.Path(categories).expanduser().resolve()
            categories, *_ = zip(*self.read_categories_file(path))
        self.categories = tuple(categories)

        self.citation = citation
        self.homepage = homepage
        self.license = license

    def read_categories_file(self, path: pathlib.Path) -> List[List[str]]:
        with open(path, newline="") as file:
            return [row for row in csv.reader(file)]

    def make_config(self, **options: Any) -> DatasetConfig:
        if not self.options and options:
            raise ValueError(
                f"Dataset '{self.name}' does not take any options, "
                f"but got {sequence_to_str(list(options), separate_last=' and')}."
            )

        for name, arg in options.items():
            if name not in self._options_map:
                raise ValueError(
                    add_suggestion(
                        f"Unknown option '{name}' of dataset '{self.name}'.",
                        word=name,
                        possibilities=sorted(self._options_map.keys()),
                    )
                )

            option = self._options_map[name]

            if arg not in option.valid:
                raise ValueError(f"Invalid argument '{arg}' for option '{name}' of dataset '{self.name}'. {option.doc}")

        return DatasetConfig(self.default_config, **options)

    def check_dependencies(self) -> None:
        for dependency in self.dependecies:
            try:
                importlib.import_module(dependency)
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    f"Dataset '{self.name}' depends on the third-party package '{dependency}'. "
                    f"Please install it, for example with `pip install {dependency}`."
                ) from error

    @abc.abstractmethod
    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        pass

    @abc.abstractmethod
    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        pass

    def supports_sharded(self) -> bool:
        return False

    def load(
        self,
        root: Union[str, pathlib.Path],
        *,
        config: Optional[DatasetConfig] = None,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = None,
        skip_integrity_check: bool = False,
    ) -> IterDataPipe[Dict[str, Any]]:
        if not config:
            config = self.default_config

        if use_sharded_dataset() and self.supports_sharded():
            root = os.path.join(root, *config.values())
            # TODO: since extra is no longer a thing, we need have a proper field or better yet a
            #  num_samples(config) method
            dataset_size = self.info.extra["sizes"][config]
            return _make_sharded_datapipe(root, dataset_size)  # type: ignore[no-any-return]

        self.check_dependencies()
        resource_dps = [
            resource.load(root, skip_integrity_check=skip_integrity_check) for resource in self.resources(config)
        ]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)

    def _generate_categories(self, root: pathlib.Path) -> Sequence[Union[str, Sequence[str]]]:
        raise NotImplementedError
