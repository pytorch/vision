import sys

import numpy as np
import pytest
import torch
from datasets_utils import make_fake_flo_file
from torchvision.datasets._optical_flow import _read_flo as read_flo_ref
from torchvision.prototype.datasets.utils._internal import read_flo, fromfile


@pytest.mark.filterwarnings("error:The given NumPy array is not writeable:UserWarning")
@pytest.mark.parametrize(
    ("np_dtype", "torch_dtype", "byte_order"),
    [
        (">f4", torch.float32, "big"),
        ("<f8", torch.float64, "little"),
        ("<i4", torch.int32, "little"),
        (">i8", torch.int64, "big"),
        ("|u1", torch.uint8, sys.byteorder),
    ],
)
@pytest.mark.parametrize("count", (-1, 2))
@pytest.mark.parametrize("mode", ("rb", "r+b"))
def test_fromfile(tmpdir, np_dtype, torch_dtype, byte_order, count, mode):
    path = tmpdir / "data.bin"
    rng = np.random.RandomState(0)
    rng.randn(5 if count == -1 else count + 1).astype(np_dtype).tofile(path)

    for count_ in (-1, count // 2):
        expected = torch.from_numpy(np.fromfile(path, dtype=np_dtype, count=count_).astype(np_dtype[1:]))

        with open(path, mode) as file:
            actual = fromfile(file, dtype=torch_dtype, byte_order=byte_order, count=count_)

        torch.testing.assert_close(actual, expected)


def test_read_flo(tmpdir):
    path = tmpdir / "test.flo"
    make_fake_flo_file(3, 4, path)

    with open(path, "rb") as file:
        actual = read_flo(file)

    expected = torch.from_numpy(read_flo_ref(path).astype("f4", copy=False))

    torch.testing.assert_close(actual, expected)
