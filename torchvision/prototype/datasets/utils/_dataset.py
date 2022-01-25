import abc
import csv
import enum
import importlib
import io
import itertools
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple, Collection

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.utils._internal import FrozenBunch, make_repr
from torchvision.prototype.utils._internal import add_suggestion, sequence_to_str

from .._home import use_sharded_dataset
from ._internal import BUILTIN_DIR, _make_sharded_datapipe
from ._resource import OnlineResource


class DatasetType(enum.Enum):
    ## Looks like this is only used to determine the decoder to use.
    ## Should we then have a more direct way of setting the default decoder?
    RAW = enum.auto()
    IMAGE = enum.auto()


class DatasetConfig(FrozenBunch):  ## Do we need this to be a FrozenBunch?
    pass


class DatasetInfo:
    def __init__(
        self,
        name: str,
        *,
        type: Union[str, DatasetType],
        dependencies: Collection[str] = (),
        categories: Optional[Union[int, Sequence[str], str, pathlib.Path]] = None,  ## Nit: do we need to allow Paths?
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        valid_options: Optional[Dict[str, Sequence[Any]]] = None,
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

        ## What about mutually exclusive parameters?
        ## How are we going to document the dataset parameters?
        self._valid_options = valid_options or dict()
        self._configs = tuple(
            DatasetConfig(**dict(zip(self._valid_options.keys(), combination)))
            for combination in itertools.product(*self._valid_options.values())
        )

        ## What is this?
        self.extra = FrozenBunch(extra or dict())

    @property
    def default_config(self) -> DatasetConfig:
        return self._configs[0]

    @staticmethod
    def read_categories_file(path: pathlib.Path) -> List[List[str]]:
        with open(path, newline="") as file:
            return [row for row in csv.reader(file)]  ## Nit: do we need a csv reader?

    def make_config(self, **options: Any) -> DatasetConfig:
        if not self._valid_options and options:
            raise ValueError(
                f"Dataset {self.name} does not take any options, "
                f"but got {sequence_to_str(list(options), separate_last=' and')}."
            )

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

    ## Could these @properties be simple attributes?
    ## Also, do we actually need to expose these attributes in the Dataset class, or could we just access dataset.info.the_attr ?
    ## Dumb idea: what if we get rid of the `info` attribute and put everything in the Dataset class namespace?
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

    ## Q: why is resources() "public" while _make_datapipe() isn't?
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

    ## Is this related to Manifold support or does this have OSS user facing impact?
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

        ## TODO: need to understand this bit
        if use_sharded_dataset() and self.supports_sharded():
            root = os.path.join(root, *config.values())
            dataset_size = self.info.extra["sizes"][config]
            return _make_sharded_datapipe(root, dataset_size)  # type: ignore[no-any-return]

        self.info.check_dependencies()
        ## Nit: a lot of method are called load() and they all do somewhat different things on different objects
        resource_dps = [
            resource.load(root, skip_integrity_check=skip_integrity_check) for resource in self.resources(config)
        ]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)

    def _generate_categories(self, root: pathlib.Path) -> Sequence[Union[str, Sequence[str]]]:
        raise NotImplementedError
