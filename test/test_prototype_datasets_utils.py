import sys

import numpy as np
import pytest
import torch
from datasets_utils import make_fake_flo_file
from torchvision.datasets._optical_flow import _read_flo as read_flo_ref
from torchvision.prototype.datasets.utils import HttpResource, GDriveResource
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


class TestHttpResource:
    def test_resolve_to_http(self, mocker):
        file_name = "data.tar"
        original_url = f"http://downloads.pytorch.org/{file_name}"

        redirected_url = original_url.replace("http", "https")

        sha256_sentinel = "sha256_sentinel"

        def preprocess_sentinel(path):
            return path

        original_resource = HttpResource(
            original_url,
            sha256=sha256_sentinel,
            preprocess=preprocess_sentinel,
        )

        mocker.patch("torchvision.prototype.datasets.utils._resource._get_redirect_url", return_value=redirected_url)
        redirected_resource = original_resource.resolve()

        assert isinstance(redirected_resource, HttpResource)
        assert redirected_resource.url == redirected_url
        assert redirected_resource.file_name == file_name
        assert redirected_resource.sha256 == sha256_sentinel
        assert redirected_resource._preprocess is preprocess_sentinel

    def test_resolve_to_gdrive(self, mocker):
        file_name = "data.tar"
        original_url = f"http://downloads.pytorch.org/{file_name}"

        id_sentinel = "id-sentinel"
        redirected_url = f"https://drive.google.com/file/d/{id_sentinel}/view"

        sha256_sentinel = "sha256_sentinel"

        def preprocess_sentinel(path):
            return path

        original_resource = HttpResource(
            original_url,
            sha256=sha256_sentinel,
            preprocess=preprocess_sentinel,
        )

        mocker.patch("torchvision.prototype.datasets.utils._resource._get_redirect_url", return_value=redirected_url)
        redirected_resource = original_resource.resolve()

        assert isinstance(redirected_resource, GDriveResource)
        assert redirected_resource.id == id_sentinel
        assert redirected_resource.file_name == file_name
        assert redirected_resource.sha256 == sha256_sentinel
        assert redirected_resource._preprocess is preprocess_sentinel
