import collections.abc
import csv
import difflib
import enum
import gzip
import io
import lzma
import os
import os.path
import pathlib
import textwrap
from collections import Mapping
from typing import (
    Collection,
    Sequence,
    Callable,
    Union,
    Any,
    Tuple,
    TypeVar,
    Iterator,
    Dict,
    Optional,
    NoReturn,
    Iterable,
)

import numpy as np
import PIL.Image
from torch.utils.data import IterDataPipe


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
    "make_repr",
    "FrozenMapping",
    "FrozenBunch",
    "create_categories_file",
    "read_mat",
    "image_buffer_from_array",
    "SequenceIterator",
    "MappingIterator",
    "Enumerator",
    "getitem",
    "path_accessor",
    "path_comparator",
    "Decompressor",
]

K = TypeVar("K")
D = TypeVar("D")

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if len(seq) == 1:
        return f"'{seq[0]}'"

    return f"""'{"', '".join([str(item) for item in seq[:-1]])}', """ f"""{separate_last}'{seq[-1]}'."""


def add_suggestion(
    msg: str,
    *,
    word: str,
    possibilities: Collection[str],
    close_match_hint: Callable[[str], str] = lambda close_match: f"Did you mean '{close_match}'?",
    alternative_hint: Callable[
        [Sequence[str]], str
    ] = lambda possibilities: f"Can be {sequence_to_str(possibilities, separate_last='or ')}.",
) -> str:
    if not isinstance(possibilities, collections.abc.Sequence):
        possibilities = sorted(possibilities)
    suggestions = difflib.get_close_matches(word, possibilities, 1)
    hint = close_match_hint(suggestions[0]) if suggestions else alternative_hint(possibilities)
    return f"{msg.strip()} {hint}"


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


class FrozenMapping(Mapping):
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

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __delitem__(self, key: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __hash__(self) -> int:
        return self.__dict__["__final_hash__"]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMapping):
            return NotImplemented

        return hash(self) == hash(other)

    def __repr__(self):
        return repr(self.__dict__["__data__"])


class FrozenBunch(FrozenMapping):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from error

    def __setattr__(self, key: Any, value: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __delattr__(self, item: Any) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __repr__(self) -> str:
        return make_repr(type(self).__name__, self.items())


def create_categories_file(
    root: Union[str, pathlib.Path], name: str, categories: Sequence[Union[str, Sequence[str]]], **fmtparams: Any
) -> None:
    with open(pathlib.Path(root) / f"{name}.categories", "w", newline="") as file:
        csv.writer(file, **fmtparams).writerows(categories)


def read_mat(buffer: io.IOBase, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError("Package `scipy` is required to be installed to read .mat files.") from error

    return sio.loadmat(buffer, **kwargs)


def image_buffer_from_array(array: np.ndarray, *, format: str = "png") -> io.BytesIO:
    image = PIL.Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


class SequenceIterator(IterDataPipe[D]):
    def __init__(self, datapipe: IterDataPipe[Sequence[D]]):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[D]:
        for sequence in self.datapipe:
            yield from iter(sequence)


class MappingIterator(IterDataPipe[Union[Tuple[K, D], D]]):
    def __init__(self, datapipe: IterDataPipe[Dict[K, D]], *, drop_key: bool = False) -> None:
        self.datapipe = datapipe
        self.drop_key = drop_key

    def __iter__(self) -> Iterator[Union[Tuple[K, D], D]]:
        for mapping in self.datapipe:
            yield from iter(mapping.values() if self.drop_key else mapping.items())  # type: ignore[call-overload]


class Enumerator(IterDataPipe[Tuple[int, D]]):
    def __init__(self, datapipe: IterDataPipe[D], start: int = 0) -> None:
        self.datapipe = datapipe
        self.start = start

    def __iter__(self) -> Iterator[Tuple[int, D]]:
        yield from enumerate(self.datapipe, self.start)


def getitem(*items: Any) -> Callable[[Any], Any]:
    def wrapper(obj: Any):
        for item in items:
            obj = obj[item]
        return obj

    return wrapper


def path_accessor(getter: Union[str, Callable[[pathlib.Path], D]]) -> Callable[[Tuple[str, Any]], D]:
    if isinstance(getter, str):
        name = getter

        def getter(path: pathlib.Path) -> D:
            return getattr(path, name)

    def wrapper(data: Tuple[str, Any]) -> D:
        return getter(pathlib.Path(data[0]))  # type: ignore[operator]

    return wrapper


def path_comparator(getter: Union[str, Callable[[pathlib.Path], D]], value: D) -> Callable[[Tuple[str, Any]], bool]:
    accessor = path_accessor(getter)

    def wrapper(data: Tuple[str, Any]) -> bool:
        return accessor(data) == value

    return wrapper


class CompressionType(enum.Enum):
    GZIP = "gzip"
    LZMA = "lzma"


class Decompressor(IterDataPipe[Tuple[str, io.IOBase]]):
    types = CompressionType

    _DECOMPRESSORS = {
        types.GZIP: lambda file: gzip.GzipFile(fileobj=file),
        types.LZMA: lambda file: lzma.LZMAFile(file),
    }

    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, io.IOBase]],
        *,
        type: Optional[Union[str, CompressionType]] = None,
    ) -> None:
        self.datapipe = datapipe
        if isinstance(type, str):
            type = self.types(type.upper())
        self.type = type

    def _detect_compression_type(self, path: str) -> CompressionType:
        if self.type:
            return self.type

        # TODO: this needs to be more elaborate
        ext = os.path.splitext(path)[1]
        if ext == ".gz":
            return self.types.GZIP
        elif ext == ".xz":
            return self.types.LZMA
        else:
            raise RuntimeError("FIXME")

    def __iter__(self) -> Iterator[Tuple[str, io.IOBase]]:
        for path, file in self.datapipe:
            type = self._detect_compression_type(path)
            decompressor = self._DECOMPRESSORS[type]
            yield path, decompressor(file)
