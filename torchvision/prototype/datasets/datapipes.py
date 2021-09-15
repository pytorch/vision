"""This is a temporary module for datapipes that should be absorbed by torchdata"""

import enum
import gzip
import io
import itertools
import lzma
import os.path
from typing import Dict, Iterator, Optional, Sequence, Tuple, TypeVar, Union

from torch.utils.data import IterDataPipe

__all__ = [
    "Decompressor",
    "Enumerator",
    "Slicer",
    "SequenceIterator",
    "MappingIterator",
]


T = TypeVar("T")


class CompressionType(enum.Enum):
    GZIP = "gzip"
    LZMA = "lzma"


class Decompressor(IterDataPipe):
    types = CompressionType

    _DECOMPRESSORS = {
        types.GZIP: lambda file: gzip.GzipFile(fileobj=file),
        types.LZMA: lambda file: lzma.LZMAFile(file),
    }

    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, io.BufferedIOBase]],
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

    def __iter__(self):
        for path, file in self.datapipe:
            type = self._detect_compression_type(path)
            decompressor = self._DECOMPRESSORS[type]
            yield path, decompressor(file)


class Enumerator(IterDataPipe[T]):
    def __init__(self, datapipe: IterDataPipe[T], *, start: int = 0) -> None:
        self.datapipe = datapipe
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"start must be a non-negative integer, but got {start}.")
        self.start = start

    def __iter__(self) -> Iterator[Tuple[int, T]]:
        yield from enumerate(self.datapipe, self.start)


class Slicer(IterDataPipe[T]):
    def __init__(
        self,
        datapipe: IterDataPipe[T],
        *,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self.datapipe = datapipe
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self) -> Iterator[T]:
        yield from itertools.islice(self.datapipe, self.start, self.stop, self.step)


class SequenceIterator(IterDataPipe[T]):
    def __init__(self, datapipe: IterDataPipe[Sequence[T]]):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[T]:
        for sequence in self.datapipe:
            yield from iter(sequence)


class MappingIterator(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict], *, drop_key: bool = False) -> None:
        self.datapipe = datapipe
        self.drop_key = drop_key

    def __iter__(self) -> Iterator[Tuple]:
        for mapping in self.datapipe:
            yield from iter(mapping.values() if self.drop_key else mapping.items())
