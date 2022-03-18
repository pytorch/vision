import collections.abc
import difflib
import functools
import inspect
import io
import mmap
import os
import os.path
import platform
import textwrap
import warnings
from typing import (
    Any,
    BinaryIO,
    Callable,
    cast,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Optional,
)

import numpy as np
import torch
from torchvision._utils import sequence_to_str


__all__ = [
    "add_suggestion",
    "FrozenMapping",
    "make_repr",
    "FrozenBunch",
    "kwonly_to_pos_or_kw",
    "fromfile",
    "ReadOnlyTensorBuffer",
    "apply_recursively",
    "query_recursively",
]


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
    if not hint:
        return msg

    return f"{msg.strip()} {hint}"


K = TypeVar("K")
D = TypeVar("D")


class FrozenMapping(Mapping[K, D]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data = dict(*args, **kwargs)
        self.__dict__["__data__"] = data
        self.__dict__["__final_hash__"] = hash(tuple(data.items()))

    def __getitem__(self, item: K) -> D:
        return cast(Mapping[K, D], self.__dict__["__data__"])[item]

    def __iter__(self) -> Iterator[K]:
        return iter(self.__dict__["__data__"].keys())

    def __len__(self) -> int:
        return len(self.__dict__["__data__"])

    def __immutable__(self) -> NoReturn:
        raise RuntimeError(f"'{type(self).__name__}' object is immutable")

    def __setitem__(self, key: K, value: Any) -> NoReturn:
        self.__immutable__()

    def __delitem__(self, key: K) -> NoReturn:
        self.__immutable__()

    def __hash__(self) -> int:
        return cast(int, self.__dict__["__final_hash__"])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMapping):
            return NotImplemented

        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return repr(self.__dict__["__data__"])


def make_repr(name: str, items: Iterable[Tuple[str, Any]]) -> str:
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


class FrozenBunch(FrozenMapping):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from error

    def __setattr__(self, key: Any, value: Any) -> NoReturn:
        self.__immutable__()

    def __delattr__(self, item: Any) -> NoReturn:
        self.__immutable__()

    def __repr__(self) -> str:
        return make_repr(type(self).__name__, self.items())


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated. Please use keyword parameter(s) instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper


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
        except (AttributeError, PermissionError, io.UnsupportedOperation):
            buffer = _read_mutable_buffer_fallback(file, count, item_size)
    else:
        # On Windows just trying to call mmap.mmap() on a file that does not support it, may corrupt the internal state
        # so no data can be read afterwards. Thus, we simply ignore the possible speed-up.
        buffer = _read_mutable_buffer_fallback(file, count, item_size)

    # We cannot use torch.frombuffer() directly, since it only supports the native byte order of the system. Thus, we
    # read the data with np.frombuffer() with the correct byte order and convert it to the native one with the
    # successive .astype() call.
    return torch.from_numpy(np.frombuffer(buffer, dtype=np_dtype, count=count).astype(np_dtype[1:], copy=False))


class ReadOnlyTensorBuffer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._memory = memoryview(tensor.numpy())
        self._cursor: int = 0

    def tell(self) -> int:
        return self._cursor

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._cursor = offset
        elif whence == io.SEEK_CUR:
            self._cursor += offset
            pass
        elif whence == io.SEEK_END:
            self._cursor = len(self._memory) + offset
        else:
            raise ValueError(
                f"'whence' should be ``{io.SEEK_SET}``, ``{io.SEEK_CUR}``, or ``{io.SEEK_END}``, "
                f"but got {repr(whence)} instead"
            )
        return self.tell()

    def read(self, size: int = -1) -> bytes:
        cursor = self.tell()
        offset, whence = (0, io.SEEK_END) if size == -1 else (size, io.SEEK_CUR)
        return self._memory[slice(cursor, self.seek(offset, whence))].tobytes()


def apply_recursively(fn: Callable, obj: Any) -> Any:
    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return [apply_recursively(fn, item) for item in obj]
    elif isinstance(obj, collections.abc.Mapping):
        return {key: apply_recursively(fn, item) for key, item in obj.items()}
    else:
        return fn(obj)


def query_recursively(
    fn: Callable[[Tuple[Any, ...], Any], Optional[D]], obj: Any, *, id: Tuple[Any, ...] = ()
) -> Iterator[D]:
    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        for idx, item in enumerate(obj):
            yield from query_recursively(fn, item, id=(*id, idx))
    elif isinstance(obj, collections.abc.Mapping):
        for key, item in obj.items():
            yield from query_recursively(fn, item, id=(*id, key))
    else:
        result = fn(id, obj)
        if result is not None:
            yield result
