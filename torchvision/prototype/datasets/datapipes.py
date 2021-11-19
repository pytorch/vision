from typing import Tuple, IO, Iterator, Union, cast

import torch
from torchdata.datapipes.iter import IterDataPipe

__all__ = ["FloReader"]


class FloReader(IterDataPipe[torch.Tensor]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, IO]]) -> None:
        self.datapipe = datapipe

    def _read_data(self, file: IO, *, dtype: torch.dtype, count: int) -> torch.Tensor:
        num_bytes_per_value = (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits // 8
        chunk_size = count * num_bytes_per_value
        return torch.frombuffer(bytearray(file.read(chunk_size)), dtype=dtype)

    def _read_scalar(self, file: IO, *, dtype: torch.dtype) -> Union[int, float]:
        return self._read_data(file, dtype=dtype, count=1).item()

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _, file in self.datapipe:
            if self._read_scalar(file, dtype=torch.float32) != 202021.25:
                raise ValueError("Magic number incorrect. Invalid .flo file")

            width = cast(int, self._read_scalar(file, dtype=torch.int32))
            height = cast(int, self._read_scalar(file, dtype=torch.int32))

            yield self._read_data(file, dtype=torch.float32, count=2 * height * width).reshape((2, height, width))
