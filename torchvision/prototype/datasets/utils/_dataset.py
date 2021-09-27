import abc
import io
import os
import pathlib
import textwrap
from collections import Mapping
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    NoReturn,
    Iterable,
    Tuple,
)

import torch
from torch.utils.data import IterDataPipe

from torchvision.prototype.datasets.utils._internal import (
    add_suggestion,
    sequence_to_str,
)
from ._resource import OnlineResource


def make_repr(name: str, items: Iterable[Tuple[str, Any]]):
    def to_str(sep: str) -> str:
        return sep.join([f"{key}={value}" for key, value in items])

    prefix = f"{name}("
    postfix = ")"
    body = to_str(", ")

    line_length = int(os.environ.get("COLUMNS", 80))
    body_too_long = (len(prefix) + len(body) + len(postfix)) > line_length
    multiline_body = len(str(body).splitlines()) > 1
    if not (body_too_long or multiline_body):
        return prefix + body + postfix

    body = textwrap.indent(to_str(",\n"), " " * 2)
    return f"{prefix}\n{body}\n{postfix}"


class DatasetConfig(Mapping):
    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        self.__dict__["__data__"] = data
        self.__dict__["__final_hash__"] = hash(tuple(data.items()))

    def __getitem__(self, name: str) -> Any:
        return self.__dict__["__data__"][name]

    def __iter__(self):
        return iter(self.__dict__["__data__"].keys())

    def __len__(self):
        return len(self.__dict__["__data__"])

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from error

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __setattr__(self, key: Any, value: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __delitem__(self, key: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __delattr__(self, item: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __hash__(self) -> int:
        return self.__dict__["__final_hash__"]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DatasetConfig):
            return NotImplemented

        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return make_repr(type(self).__name__, self.items())


class DatasetInfo:
    def __init__(
        self,
        name: str,
        *,
        categories: Union[int, Sequence[str], str, pathlib.Path],
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        valid_options: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        self.name = name.lower()

        if isinstance(categories, int):
            categories = [str(label) for label in range(categories)]
        elif isinstance(categories, (str, pathlib.Path)):
            with open(pathlib.Path(categories).expanduser().resolve(), "r") as fh:
                categories = fh.readlines()
        self.categories = categories

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

    @property
    def default_config(self) -> DatasetConfig:
        return DatasetConfig(
            {name: valid_args[0] for name, valid_args in self._valid_options.items()}
        )

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
        items.extend(
            sorted(
                (key, sequence_to_str(value))
                for key, value in self._valid_options.items()
            )
        )
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

        resource_dps = [
            resource.to_datapipe(root) for resource in self.resources(config)
        ]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)
