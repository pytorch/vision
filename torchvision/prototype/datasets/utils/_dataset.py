import abc
import io
import os
import pathlib
import textwrap
from collections import UserDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import IterDataPipe

from ._resource import OnlineResource

__all__ = ["DatasetConfig", "DatasetInfo", "Dataset"]


class DatasetConfig(UserDict):
    _INDENT = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: why do we require this again?
        # DatasetConfig should be immutable so it should be hashable
        self.__dict__["__final_hash__"] = hash(tuple(self.items()))

    def __hash__(self) -> int:
        return self.__dict__["__final_hash__"]

    def __getattr__(self, name: Any) -> Any:
        if name == "data":
            return self.__dict__[name]

        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: Any, value: Any) -> None:
        if name == "data":
            self.__dict__[name] = value
            return

        # TODO Shouldn't this be disabled if this immutable?

        self.data[name] = value

    def __repr__(self):
        def to_str(sep: str) -> str:
            return sep.join([f"{key}={value}" for key, value in self.items()])

        prefix = f"{type(self).__name__}("
        postfix = ")"
        body = to_str(", ")

        line_length = int(os.environ.get("COLUMNS", 80))
        body_too_long = (len(prefix) + len(body) + len(postfix)) > line_length
        multiline_body = len(str(body).splitlines()) > 1
        if not (body_too_long or multiline_body):
            return prefix + body + postfix

        body = textwrap.indent(to_str(",\n"), " " * self._INDENT)
        return f"{prefix}\n{body}\n{postfix}"


class DatasetInfo:
    def __init__(
        self,
        name: str,
        *,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        options: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.name = name.lower()
        self.citation = citation
        self.homepage = homepage
        self.license = license
        if options is None:
            options = dict(split=("train",))
        self.options = options

    @property
    def default_config(self) -> DatasetConfig:
        return DatasetConfig({name: valid_args[0] for name, valid_args in self.options.items()})

    def make_config(self, **options: Any) -> DatasetConfig:
        for name, arg in options.items():
            if name not in self.options:
                raise ValueError

            valid_args = self.options[name]

            if arg not in valid_args:
                raise ValueError

        return DatasetConfig(self.default_config, **options)


class Dataset(abc.ABC):
    @property
    @abc.abstractmethod
    def info(self) -> DatasetInfo:
        pass

    @abc.abstractmethod
    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        pass

    @abc.abstractmethod
    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]],
    ) -> IterDataPipe[Dict[str, Any]]:
        pass

    def to_datapipe(
        self,
        root: Union[str, pathlib.Path],
        *,
        config: Optional[DatasetConfig] = None,
        decoder: Optional[Callable[[str, io.BufferedIOBase], torch.Tensor]] = None,
    ) -> IterDataPipe[Dict[str, Any]]:
        if not config:
            config = self.info.default_config

        resource_dps = [resource.to_datapipe(root) for resource in self.resources(config)]
        return self._make_datapipe(resource_dps, config=config, decoder=decoder)
