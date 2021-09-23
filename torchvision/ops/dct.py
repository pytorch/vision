import typing

import torch
import torch.fft


def dct(
        x: torch.Tensor,
        type: int = 2,
        n: int = None,
        axis: int = -1,
        norm: typing.Optional[str] = None
) -> torch.Tensor:
    if type not in (1, 2, 3, 4):
        raise ValueError

    if type == 1:
        if norm == "ortho":
            raise ValueError

        if x.shape[-1] is not None and x.shape[-1] < 2:
            raise ValueError

    if n is not None and n < 1:
        raise ValueError

    if axis != -1:
        raise NotImplementedError

    if norm not in (None, "ortho"):
        raise ValueError

    if n is not None:
        x = _pad(x, n)

    if type == 1:
        return _dct_type_1(x)

    if type == 2:
        return _dct_type_2(x, norm)

    if type == 3:
        return _dct_type_3(x, norm)

    if type == 4:
        return _dct_type_4(x, axis, norm)


def _dct_type_1(x: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.fft.rfft(torch.cat([x, torch.flip(x, [-1])[..., 1:-1:1]], -1)))


def _dct_type_2(x: torch.Tensor, norm: typing.Optional[str] = None) -> torch.Tensor:
    raise NotImplementedError


def _dct_type_3(x: torch.Tensor, norm: typing.Optional[str] = None) -> torch.Tensor:
    raise NotImplementedError


def _dct_type_4(x: torch.Tensor, axis: int = -1, norm: typing.Optional[str] = None) -> torch.Tensor:
    raise NotImplementedError


def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
    raise NotImplementedError
