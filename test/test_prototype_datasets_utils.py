import pytest
import torch
from datasets_utils import make_fake_flo_file
from torchvision.datasets._optical_flow import _read_flo as read_flo_ref
from torchvision.prototype.datasets.utils._internal import read_flo


@pytest.mark.parametrize("mode", ("rb", "r+b"))
def test_read_flo(tmpdir, mode):
    path = tmpdir / "test.flo"
    height, width = torch.randint(3, 10, (2,))
    make_fake_flo_file(height, width, path)

    with open(path, mode) as file:
        actual = read_flo(file)

    expected = torch.from_numpy(read_flo_ref(path).astype("f4", copy=False))

    torch.testing.assert_close(actual, expected)
