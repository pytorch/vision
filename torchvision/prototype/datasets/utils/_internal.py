import collections.abc
import difflib
import enum
import gzip
import io
import itertools
import lzma
import os.path
import pathlib
from typing import Collection, Sequence, Callable, Any, Iterator, Optional, Tuple, TypeVar, Union

import numpy as np
import PIL.Image
from torch.utils.data import IterDataPipe


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "sequence_to_str",
    "add_suggestion",
    "create_categories_file",
    "read_mat",
    "image_buffer_from_array",
    "Decompressor",
    "Slicer",
]

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000

D = TypeVar("D")


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


class Slicer(IterDataPipe[D]):
    def __init__(
        self,
        datapipe: IterDataPipe[D],
        *,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self.datapipe = datapipe
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self) -> Iterator[D]:
        yield from itertools.islice(self.datapipe, self.start, self.stop, self.step)
