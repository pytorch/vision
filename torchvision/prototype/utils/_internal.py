import collections.abc
import difflib
import io
import mmap
import platform
from typing import BinaryIO, Callable, Collection, Sequence, TypeVar, Union

import numpy as np
import torch
from torchvision._utils import sequence_to_str


__all__ = [
    "add_suggestion",
    "fromfile",
    "ReadOnlyTensorBuffer",
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


D = TypeVar("D")


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
        self._memory = memoryview(tensor.numpy())  # type: ignore[arg-type]
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
