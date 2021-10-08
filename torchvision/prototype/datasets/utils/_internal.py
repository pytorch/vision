import collections.abc
import difflib
import enum
import gzip
import io
import lzma
import os.path
import pathlib
from typing import Collection, Sequence, Callable, Union, Any, TypeVar, Dict, Optional
from typing import Tuple, Iterator

import numpy as np
import PIL.Image
from torchdata.datapipes.iter import IterDataPipe


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
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
    "RarArchiveReader",
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


def create_categories_file(root: Union[str, pathlib.Path], name: str, categories: Sequence[str]) -> None:
    with open(pathlib.Path(root) / f"{name}.categories", "w") as fh:
        fh.write("\n".join(categories) + "\n")


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


class RarArchiveReader(IterDataPipe[Tuple[str, io.BufferedIOBase]]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, io.BufferedIOBase]]):
        self._rarfile = self._verify_dependencies()
        super().__init__()
        self.datapipe = datapipe

    @staticmethod
    def _verify_dependencies():
        try:
            import rarfile
        except ImportError as error:
            raise ModuleNotFoundError(
                "Package `rarfile` is required to be installed to use this datapipe. "
                "Please use `pip install rarfile` or `conda -c conda-forge install rarfile` to install it."
            ) from error

        # check if at least one system library for reading rar archives is available to be used by rarfile
        rarfile.tool_setup()

        return rarfile

    def __iter__(self) -> Iterator[Tuple[str, io.BufferedIOBase]]:
        for path, stream in self.datapipe:
            rar = self._rarfile.RarFile(stream)
            for info in rar.infolist():
                if info.filename.endswith("/"):
                    continue

                inner_path = os.path.join(path, info.filename)
                file_obj = rar.open(info)

                yield inner_path, file_obj
