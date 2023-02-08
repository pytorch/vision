import abc
import importlib
import pathlib
from typing import Any, Collection, Dict, Iterator, List, Optional, Sequence, Union

from torchdata.datapipes.iter import IterDataPipe
from torchvision.datasets.utils import verify_str_arg

from ._resource import OnlineResource


class Dataset(IterDataPipe[Dict[str, Any]], abc.ABC):
    @staticmethod
    def _verify_str_arg(
        value: str,
        arg: Optional[str] = None,
        valid_values: Optional[Collection[str]] = None,
        *,
        custom_msg: Optional[str] = None,
    ) -> str:
        return verify_str_arg(value, arg, valid_values, custom_msg=custom_msg)

    def __init__(
        self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False, dependencies: Collection[str] = ()
    ) -> None:
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"{type(self).__name__}() depends on the third-party package '{dependency}'. "
                    f"Please install it, for example with `pip install {dependency}`."
                ) from None

        self._root = pathlib.Path(root).expanduser().resolve()
        resources = [
            resource.load(self._root, skip_integrity_check=skip_integrity_check) for resource in self._resources()
        ]
        self._dp = self._datapipe(resources)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._dp

    @abc.abstractmethod
    def _resources(self) -> List[OnlineResource]:
        pass

    @abc.abstractmethod
    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def _generate_categories(self) -> Sequence[Union[str, Sequence[str]]]:
        raise NotImplementedError
