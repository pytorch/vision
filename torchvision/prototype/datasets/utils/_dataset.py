import abc
import enum
import io
import pathlib
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    Tuple,
)

import torch
from torch.utils.data import IterDataPipe
from torchvision.prototype.datasets.utils._internal import (
    add_suggestion,
    sequence_to_str,
)

from ._internal import FrozenBunch, make_repr
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
        categories: Optional[Union[int, Sequence[str], str, pathlib.Path]] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        valid_options: Optional[Dict[str, Sequence]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name.lower()
        self.type = DatasetType[type.upper()] if isinstance(type, str) else type

        if categories is None:
            categories = []
        elif isinstance(categories, int):
            categories = [str(label) for label in range(categories)]
        elif isinstance(categories, (str, pathlib.Path)):
            categories = self._read_categories_file(pathlib.Path(categories).expanduser().resolve())
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

        self.extra = FrozenBunch(extra or dict())

    @staticmethod
    def _read_categories_file(path: pathlib.Path) -> List[str]:
        if not path.exists() or not path.is_file():
            warnings.warn(
                f"The categories file {path} does not exist. Continuing without loaded categories.", UserWarning
            )
            return []

        with open(path, "r") as file:
            return [line.strip() for line in file]

    @property
    def default_config(self) -> DatasetConfig:
        return DatasetConfig({name: valid_args[0] for name, valid_args in self._valid_options.items()})

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

    def __repr__(self) -> str:
        items = [("name", self.name)]
        for key in ("citation", "homepage", "license"):
            value = getattr(self, key)
            if value is not None:
                items.append((key, value))
        items.extend(sorted((key, sequence_to_str(value)) for key, value in self._valid_options.items()))
        return make_repr(type(self).__name__, items)


class Dataset(abc.ABC):
    @property
    @abc.abstractmethod
    def info(self) -> DatasetInfo:
        pass

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

    def to_datapipe(
        self,
        root: Union[str, pathlib.Path],
        *,
        config: Optional[DatasetConfig] = None,
        decoder: Optional[Callable[[io.IOBase], torch.Tensor]] = None,
    ) -> IterDataPipe[Dict[str, Any]]:
        if not config:
            config = self.info.default_config

        resource_dps = [resource.to_datapipe(root) for resource in self.resources(config)]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)

    def _generate_categories(self, root: pathlib.Path) -> Sequence[str]:
        raise NotImplementedError
