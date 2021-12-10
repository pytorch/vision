import abc
import csv
import enum
import importlib
import io
import itertools
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.utils._internal import FrozenBunch, make_repr
from torchvision.prototype.utils._internal import add_suggestion, sequence_to_str

from .._home import use_sharded_dataset
from ._internal import BUILTIN_DIR, _make_sharded_datapipe
from ._resource import OnlineResource


class DatasetType(enum.Enum):
    RAW = enum.auto()
    IMAGE = enum.auto()


class DatasetConfig(FrozenBunch):
    pass


class DatasetInfo:
    def __init__(
        self,
        name: str,
        *,
        type: Union[str, DatasetType],
        dependencies: Sequence[str] = (),
        categories: Optional[Union[int, Sequence[str], str, pathlib.Path]] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        valid_options: Optional[Dict[str, Sequence]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name.lower()
        self.type = DatasetType[type.upper()] if isinstance(type, str) else type

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

        valid_split: Dict[str, Sequence] = dict(split=["train"])
        if valid_options is None:
            valid_options = valid_split
        elif "split" not in valid_options:
            valid_options.update(valid_split)
        elif "train" not in valid_options["split"]:
            raise ValueError(
                f"'train' has to be a valid argument for option 'split', "
                f"but found only {sequence_to_str(valid_options['split'], separate_last='and ')}."
            )
        self._valid_options: Dict[str, Sequence] = valid_options
        self._configs = tuple(
            DatasetConfig(**dict(zip(valid_options.keys(), combination)))
            for combination in itertools.product(*valid_options.values())
        )

        self.extra = FrozenBunch(extra or dict())

    @property
    def default_config(self) -> DatasetConfig:
        return self._configs[0]

    @staticmethod
    def read_categories_file(path: pathlib.Path) -> List[List[str]]:
        with open(path, newline="") as file:
            return [row for row in csv.reader(file)]

    def make_config(self, **options: Any) -> DatasetConfig:
        for name, arg in options.items():
            if name not in self._valid_options:
                raise ValueError(
                    add_suggestion(
                        f"Unknown option '{name}' of dataset {self.name}.",
                        word=name,
                        possibilities=sorted(self._valid_options.keys()),
                    )
                )

            valid_args = self._valid_options[name]

            if arg not in valid_args:
                raise ValueError(
                    add_suggestion(
                        f"Invalid argument '{arg}' for option '{name}' of dataset {self.name}.",
                        word=arg,
                        possibilities=valid_args,
                    )
                )

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

    def __repr__(self) -> str:
        items = [("name", self.name)]
        for key in ("citation", "homepage", "license"):
            value = getattr(self, key)
            if value is not None:
                items.append((key, value))
        items.extend(sorted((key, sequence_to_str(value)) for key, value in self._valid_options.items()))
        return make_repr(type(self).__name__, items)


class Dataset(abc.ABC):
    def __init__(self) -> None:
        self._info = self._make_info()

    @abc.abstractmethod
    def _make_info(self) -> DatasetInfo:
        pass

    @property
    def info(self) -> DatasetInfo:
        return self._info

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def default_config(self) -> DatasetConfig:
        return self.info.default_config

    @property
    def categories(self) -> Tuple[str, ...]:
        return self.info.categories

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
            config = self.info.default_config

        if use_sharded_dataset() and self.supports_sharded():
            root = os.path.join(root, *config.values())
            dataset_size = self.info.extra["sizes"][config]
            return _make_sharded_datapipe(root, dataset_size)

        self.info.check_dependencies()
        resource_dps = [
            resource.load(root, skip_integrity_check=skip_integrity_check) for resource in self.resources(config)
        ]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)

    def _generate_categories(self, root: pathlib.Path) -> Sequence[Union[str, Sequence[str]]]:
        raise NotImplementedError
