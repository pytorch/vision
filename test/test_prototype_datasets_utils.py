import gzip
import pathlib
import sys

import numpy as np
import pytest
import torch
from datasets_utils import make_fake_flo_file, make_tar
from torchdata.datapipes.iter import FileOpener, TarArchiveLoader
from torchvision.datasets._optical_flow import _read_flo as read_flo_ref
from torchvision.datasets.utils import _decompress
from torchvision.prototype.datasets.utils import HttpResource, GDriveResource, Dataset, OnlineResource
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


class TestOnlineResource:
    class DummyResource(OnlineResource):
        def __init__(self, download_fn=None, **kwargs):
            super().__init__(**kwargs)
            self._download_fn = download_fn

        def _download(self, root):
            if self._download_fn is None:
                raise pytest.UsageError(
                    "`_download()` was called, but `DummyResource(...)` was constructed without `download_fn`."
                )

            return self._download_fn(self, root)

    def _make_file(self, root, *, content, name="file.txt"):
        file = root / name
        with open(file, "w") as fh:
            fh.write(content)

        return file

    def _make_folder(self, root, *, name="folder"):
        folder = root / name
        subfolder = folder / "subfolder"
        subfolder.mkdir(parents=True)

        files = {}
        for idx, root in enumerate([folder, folder, subfolder]):
            content = f"sentinel{idx}"
            file = self._make_file(root, name=f"file{idx}.txt", content=content)
            files[str(file)] = content

        return folder, files

    def _make_tar(self, root, *, name="archive.tar", remove=True):
        folder, files = self._make_folder(root, name=name.split(".")[0])
        archive = make_tar(root, name, folder, remove=remove)
        files = {str(archive / pathlib.Path(file).relative_to(root)): content for file, content in files.items()}
        return archive, files

    def test_load_file(self, tmp_path):
        content = "sentinel"
        file = self._make_file(tmp_path, content=content)

        resource = self.DummyResource(file_name=file.name)

        dp = resource.load(tmp_path)
        assert isinstance(dp, FileOpener)

        data = list(dp)
        assert len(data) == 1

        path, buffer = data[0]
        assert path == str(file)
        assert buffer.read().decode() == content

    def test_load_folder(self, tmp_path):
        folder, files = self._make_folder(tmp_path)

        resource = self.DummyResource(file_name=folder.name)

        dp = resource.load(tmp_path)
        assert isinstance(dp, FileOpener)
        assert {path: buffer.read().decode() for path, buffer in dp} == files

    def test_load_archive(self, tmp_path):
        archive, files = self._make_tar(tmp_path)

        resource = self.DummyResource(file_name=archive.name)

        dp = resource.load(tmp_path)
        assert isinstance(dp, TarArchiveLoader)
        assert {path: buffer.read().decode() for path, buffer in dp} == files

    def test_priority_decompressed_gt_raw(self, tmp_path):
        # We don't need to actually compress here. Adding the suffix is sufficient
        self._make_file(tmp_path, content="raw_sentinel", name="file.txt.gz")
        file = self._make_file(tmp_path, content="decompressed_sentinel", name="file.txt")

        resource = self.DummyResource(file_name=file.name)

        dp = resource.load(tmp_path)
        path, buffer = next(iter(dp))

        assert path == str(file)
        assert buffer.read().decode() == "decompressed_sentinel"

    def test_priority_extracted_gt_decompressed(self, tmp_path):
        archive, _ = self._make_tar(tmp_path, remove=False)

        resource = self.DummyResource(file_name=archive.name)

        dp = resource.load(tmp_path)
        # If the archive had been selected, this would be a `TarArchiveReader`
        assert isinstance(dp, FileOpener)

    def test_download(self, tmp_path):
        download_fn_was_called = False

        def download_fn(resource, root):
            nonlocal download_fn_was_called
            download_fn_was_called = True

            return self._make_file(root, content="_", name=resource.file_name)

        resource = self.DummyResource(
            file_name="file.txt",
            download_fn=download_fn,
        )

        resource.load(tmp_path)

        assert download_fn_was_called, "`download_fn()` was never called"

    # This tests the `"decompress"` literal as well as a custom callable
    @pytest.mark.parametrize(
        "preprocess",
        [
            "decompress",
            lambda path: _decompress(str(path), remove_finished=True),
        ],
    )
    def test_preprocess_decompress(self, tmp_path, preprocess):
        file_name = "file.txt.gz"
        content = "sentinel"

        def download_fn(resource, root):
            file = root / resource.file_name
            with gzip.open(file, "wb") as fh:
                fh.write(content.encode())
            return file

        resource = self.DummyResource(file_name=file_name, preprocess=preprocess, download_fn=download_fn)

        dp = resource.load(tmp_path)
        data = list(dp)
        assert len(data) == 1

        path, buffer = data[0]
        assert path == str(tmp_path / file_name).replace(".gz", "")
        assert buffer.read().decode() == content

    def test_preprocess_extract(self, tmp_path):
        files = None

        def download_fn(resource, root):
            nonlocal files
            archive, files = self._make_tar(root, name=resource.file_name)
            return archive

        resource = self.DummyResource(file_name="folder.tar", preprocess="extract", download_fn=download_fn)

        dp = resource.load(tmp_path)
        assert files is not None, "`download_fn()` was never called"
        assert isinstance(dp, FileOpener)

        actual = {path: buffer.read().decode() for path, buffer in dp}
        expected = {
            path.replace(resource.file_name, resource.file_name.split(".")[0]): content
            for path, content in files.items()
        }
        assert actual == expected

    def test_preprocess_only_after_download(self, tmp_path):
        file = self._make_file(tmp_path, content="_")

        def preprocess(path):
            raise AssertionError("`preprocess` was called although the file was already present.")

        resource = self.DummyResource(
            file_name=file.name,
            preprocess=preprocess,
        )

        resource.load(tmp_path)


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


def test_missing_dependency_error():
    class DummyDataset(Dataset):
        def __init__(self):
            super().__init__(root="root", dependencies=("fake_dependency",))

        def _resources(self):
            pass

        def _datapipe(self, resource_dps):
            pass

        def __len__(self):
            pass

    with pytest.raises(ModuleNotFoundError, match="depends on the third-party package 'fake_dependency'"):
        DummyDataset()
