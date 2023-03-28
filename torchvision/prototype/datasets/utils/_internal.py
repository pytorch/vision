import csv
import functools
import pathlib
import pickle
from typing import Any, BinaryIO, Callable, Dict, IO, Iterator, List, Sequence, Sized, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
import torch.utils.data
from torchdata.datapipes.iter import IoPathFileLister, IoPathFileOpener, IterDataPipe, ShardingFilter, Shuffler
from torchvision.prototype.utils._internal import fromfile


__all__ = [
    "INFINITE_BUFFER_SIZE",
    "BUILTIN_DIR",
    "read_mat",
    "MappingIterator",
    "getitem",
    "path_accessor",
    "path_comparator",
    "read_flo",
    "hint_sharding",
    "hint_shuffling",
]

K = TypeVar("K")
D = TypeVar("D")

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000

BUILTIN_DIR = pathlib.Path(__file__).parent.parent / "_builtin"


def read_mat(buffer: BinaryIO, **kwargs: Any) -> Any:
    try:
        import scipy.io as sio
    except ImportError as error:
        raise ModuleNotFoundError("Package `scipy` is required to be installed to read .mat files.") from error

    data = sio.loadmat(buffer, **kwargs)
    buffer.close()
    return data


class MappingIterator(IterDataPipe[Union[Tuple[K, D], D]]):
    def __init__(self, datapipe: IterDataPipe[Dict[K, D]], *, drop_key: bool = False) -> None:
        self.datapipe = datapipe
        self.drop_key = drop_key

    def __iter__(self) -> Iterator[Union[Tuple[K, D], D]]:
        for mapping in self.datapipe:
            yield from iter(mapping.values() if self.drop_key else mapping.items())


def _getitem_closure(obj: Any, *, items: Sequence[Any]) -> Any:
    for item in items:
        obj = obj[item]
    return obj


def getitem(*items: Any) -> Callable[[Any], Any]:
    return functools.partial(_getitem_closure, items=items)


def _getattr_closure(obj: Any, *, attrs: Sequence[str]) -> Any:
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _path_attribute_accessor(path: pathlib.Path, *, name: str) -> Any:
    return _getattr_closure(path, attrs=name.split("."))


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


class PicklerDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO[bytes]]]) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Iterator[Any]:
        for _, fobj in self.source_datapipe:
            data = pickle.load(fobj)
            for _, d in enumerate(data):
                yield d


class SharderDataPipe(ShardingFilter):
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


def _make_sharded_datapipe(root: str, dataset_size: int) -> IterDataPipe[Dict[str, Any]]:
    dp = IoPathFileLister(root=root)
    dp = SharderDataPipe(dp)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)
    dp = IoPathFileOpener(dp, mode="rb")
    dp = PicklerDataPipe(dp)
    # dp = dp.cycle(2)
    dp = TakerDataPipe(dp, dataset_size)
    return dp


def read_flo(file: BinaryIO) -> torch.Tensor:
    if file.read(4) != b"PIEH":
        raise ValueError("Magic number incorrect. Invalid .flo file")

    width, height = fromfile(file, dtype=torch.int32, byte_order="little", count=2)
    flow = fromfile(file, dtype=torch.float32, byte_order="little", count=height * width * 2)
    return flow.reshape((height, width, 2)).permute((2, 0, 1))


def hint_sharding(datapipe: IterDataPipe) -> ShardingFilter:
    return ShardingFilter(datapipe)


def hint_shuffling(datapipe: IterDataPipe[D]) -> Shuffler[D]:
    return Shuffler(datapipe, buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)


def read_categories_file(name: str) -> List[Union[str, Sequence[str]]]:
    path = BUILTIN_DIR / f"{name}.categories"
    with open(path, newline="") as file:
        rows = list(csv.reader(file))
        rows = [row[0] if len(row) == 1 else row for row in rows]
        return rows
