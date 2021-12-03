import enum
import functools
import gzip
import io
import lzma
import mmap
import os
import os.path
import pathlib
import pickle
import platform
from typing import BinaryIO
from typing import (
    Sequence,
    Callable,
    Union,
    Any,
    Tuple,
    TypeVar,
    Iterator,
    Dict,
    Optional,
    IO,
    Sized,
)
from typing import cast

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter import IoPathFileLister, IoPathFileLoader
from torchdata.datapipes.utils import StreamWrapper


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "BUILTIN_DIR",
    "read_mat",
    "image_buffer_from_array",
    "SequenceIterator",
    "MappingIterator",
    "Enumerator",
    "getitem",
    "path_accessor",
    "path_comparator",
    "Decompressor",
    "fromfile",
    "read_flo",
]

K = TypeVar("K")
D = TypeVar("D")

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000

BUILTIN_DIR = pathlib.Path(__file__).parent.parent / "_builtin"


def read_mat(buffer: io.IOBase, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError("Package `scipy` is required to be installed to read .mat files.") from error

    if isinstance(buffer, StreamWrapper):
        buffer = buffer.file_obj

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


def _getitem_closure(obj: Any, *, items: Tuple[Any, ...]) -> Any:
    for item in items:
        obj = obj[item]
    return obj


def getitem(*items: Any) -> Callable[[Any], Any]:
    return functools.partial(_getitem_closure, items=items)


def _path_attribute_accessor(path: pathlib.Path, *, name: str) -> D:
    return cast(D, getattr(path, name))


def _path_accessor_closure(data: Tuple[str, Any], *, getter: Callable[[pathlib.Path], D]) -> D:
    return getter(pathlib.Path(data[0]))


def path_accessor(getter: Union[str, Callable[[pathlib.Path], D]]) -> Callable[[Tuple[str, Any]], D]:
    if isinstance(getter, str):
        getter = functools.partial(_path_attribute_accessor, name=getter)

    return functools.partial(_path_accessor_closure, getter=getter)


def _path_comparator_closure(data: Tuple[str, Any], *, accessor: Callable[[Tuple[str, Any]], D], value: D) -> bool:
    return accessor(data) == value


def path_comparator(getter: Union[str, Callable[[pathlib.Path], D]], value: D) -> Callable[[Tuple[str, Any]], bool]:
    return functools.partial(_path_comparator_closure, accessor=path_accessor(getter), value=value)


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


class PicklerDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO[bytes]]]) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Iterator[Any]:
        for _, fobj in self.source_datapipe:
            data = pickle.load(fobj)
            for _, d in enumerate(data):
                yield d


class SharderDataPipe(torch.utils.data.datapipes.iter.grouping.ShardingFilterIterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        super().__init__(source_datapipe)
        self.rank = 0
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.apply_sharding(self.world_size, self.rank)

    def __iter__(self) -> Iterator[Any]:
        num_workers = self.world_size
        worker_id = self.rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_id + worker_info.id * num_workers
            num_workers *= worker_info.num_workers
        self.apply_sharding(num_workers, worker_id)
        yield from super().__iter__()


class TakerDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, num_take: int) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.num_take = num_take
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()

    def __iter__(self) -> Iterator[Any]:
        num_workers = self.world_size
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers *= worker_info.num_workers

        # TODO: this is weird as it drops more elements than it should
        num_take = self.num_take // num_workers

        for i, data in enumerate(self.source_datapipe):
            if i < num_take:
                yield data
            else:
                break

    def __len__(self) -> int:
        num_take = self.num_take // self.world_size
        if isinstance(self.source_datapipe, Sized):
            if len(self.source_datapipe) < num_take:
                num_take = len(self.source_datapipe)
        # TODO: might be weird to not take `num_workers` into account
        return num_take


def _make_sharded_datapipe(root: str, dataset_size: int) -> IterDataPipe:
    dp = IoPathFileLister(root=root)
    dp = SharderDataPipe(dp)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)
    dp = IoPathFileLoader(dp, mode="rb")
    dp = PicklerDataPipe(dp)
    # dp = dp.cycle(2)
    dp = TakerDataPipe(dp, dataset_size)
    return dp


def _read_mutable_buffer_fallback(file: BinaryIO, count: int, item_size: int) -> bytearray:
    # A plain file.read() will give a read-only bytes, so we convert it to bytearray to make it mutable
    return bytearray(file.read(-1 if count == -1 else count * item_size))


def fromfile(
    file: BinaryIO,
    *,
    dtype: torch.dtype,
    byte_order: str,
    count: int = -1,
) -> torch.Tensor:
    """Construct a tensor from a binary file.

    .. note::

        This function is similar to :func:`numpy.fromfile` with two notable differences:

        1. This function only accepts an open binary file, but not a path to it.
        2. This function has an additional ``byte_order`` parameter, since PyTorch's ``dtype``'s do not support that
            concept.

    .. note::

        If the ``file`` was opened in update mode, i.e. "r+b" or "w+b", reading data is much faster. Be aware that as
        long as the file is still open, inplace operations on the returned tensor will reflect back to the file.

    Args:
        file (IO): Open binary file.
        dtype (torch.dtype): Data type of the underlying data as well as of the returned tensor.
        byte_order (str): Byte order of the data. Can be "little" or "big" endian.
        count (int): Number of values of the returned tensor. If ``-1`` (default), will read the complete file.
    """
    byte_order = "<" if byte_order == "little" else ">"
    char = "f" if dtype.is_floating_point else ("i" if dtype.is_signed else "u")
    item_size = (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits // 8
    np_dtype = byte_order + char + str(item_size)

    buffer: Union[memoryview, bytearray]
    if platform.system() != "Windows":
        # PyTorch does not support tensors with underlying read-only memory. In case
        # - the file has a .fileno(),
        # - the file was opened for updating, i.e. 'r+b' or 'w+b',
        # - the file is seekable
        # we can avoid copying the data for performance. Otherwise we fall back to simply .read() the data and copy it
        # to a mutable location afterwards.
        try:
            buffer = memoryview(mmap.mmap(file.fileno(), 0))[file.tell() :]
            # Reading from the memoryview does not advance the file cursor, so we have to do it manually.
            file.seek(*(0, io.SEEK_END) if count == -1 else (count * item_size, io.SEEK_CUR))
        except (PermissionError, io.UnsupportedOperation):
            buffer = _read_mutable_buffer_fallback(file, count, item_size)
    else:
        # On Windows just trying to call mmap.mmap() on a file that does not support it, may corrupt the internal state
        # so no data can be read afterwards. Thus, we simply ignore the possible speed-up.
        buffer = _read_mutable_buffer_fallback(file, count, item_size)

    # We cannot use torch.frombuffer() directly, since it only supports the native byte order of the system. Thus, we
    # read the data with np.frombuffer() with the correct byte order and convert it to the native one with the
    # successive .astype() call.
    return torch.from_numpy(np.frombuffer(buffer, dtype=np_dtype, count=count).astype(np_dtype[1:], copy=False))


def read_flo(file: BinaryIO) -> torch.Tensor:
    if file.read(4) != b"PIEH":
        raise ValueError("Magic number incorrect. Invalid .flo file")

    width, height = fromfile(file, dtype=torch.int32, byte_order="little", count=2)
    flow = fromfile(file, dtype=torch.float32, byte_order="little", count=height * width * 2)
    return flow.reshape((height, width, 2)).permute((2, 0, 1))
