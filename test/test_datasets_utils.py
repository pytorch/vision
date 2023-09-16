import contextlib
import gzip
import os
import pathlib
import re
import tarfile
import zipfile

import pytest
import torch
import torchvision.datasets.utils as utils
from common_utils import assert_equal
from torch._utils_internal import get_file_path_2
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import _COMPRESSED_FILE_OPENERS

TEST_FILE = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
)


def patch_url_redirection(mocker, redirect_url):
    class Response:
        def __init__(self, url):
            self.url = url

    @contextlib.contextmanager
    def patched_opener(*args, **kwargs):
        yield Response(redirect_url)

    return mocker.patch("torchvision.datasets.utils.urllib.request.urlopen", side_effect=patched_opener)


class TestDatasetsUtils:
    def test_get_redirect_url(self, mocker):
        url = "https://url.org"
        expected_redirect_url = "https://redirect.url.org"

        mock = patch_url_redirection(mocker, expected_redirect_url)

        actual = utils._get_redirect_url(url)
        assert actual == expected_redirect_url

        assert mock.call_count == 2
        call_args_1, call_args_2 = mock.call_args_list
        assert call_args_1[0][0].full_url == url
        assert call_args_2[0][0].full_url == expected_redirect_url

    def test_get_redirect_url_max_hops_exceeded(self, mocker):
        url = "https://url.org"
        redirect_url = "https://redirect.url.org"

        mock = patch_url_redirection(mocker, redirect_url)

        with pytest.raises(RecursionError):
            utils._get_redirect_url(url, max_hops=0)

        assert mock.call_count == 1
        assert mock.call_args[0][0].full_url == url

    def test_check_md5(self):
        fpath = TEST_FILE
        correct_md5 = "9c0bb82894bb3af7f7675ef2b3b6dcdc"
        false_md5 = ""
        assert utils.check_md5(fpath, correct_md5)
        assert not utils.check_md5(fpath, false_md5)

    def test_check_integrity(self):
        existing_fpath = TEST_FILE
        nonexisting_fpath = ""
        correct_md5 = "9c0bb82894bb3af7f7675ef2b3b6dcdc"
        false_md5 = ""
        assert utils.check_integrity(existing_fpath, correct_md5)
        assert not utils.check_integrity(existing_fpath, false_md5)
        assert utils.check_integrity(existing_fpath)
        assert not utils.check_integrity(nonexisting_fpath)

    def test_get_google_drive_file_id(self):
        url = "https://drive.google.com/file/d/1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV/view"
        expected = "1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV"

        actual = utils._get_google_drive_file_id(url)
        assert actual == expected

    def test_get_google_drive_file_id_invalid_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

        assert utils._get_google_drive_file_id(url) is None

    @pytest.mark.parametrize(
        "file, expected",
        [
            ("foo.tar.bz2", (".tar.bz2", ".tar", ".bz2")),
            ("foo.tar.xz", (".tar.xz", ".tar", ".xz")),
            ("foo.tar", (".tar", ".tar", None)),
            ("foo.tar.gz", (".tar.gz", ".tar", ".gz")),
            ("foo.tbz", (".tbz", ".tar", ".bz2")),
            ("foo.tbz2", (".tbz2", ".tar", ".bz2")),
            ("foo.tgz", (".tgz", ".tar", ".gz")),
            ("foo.bz2", (".bz2", None, ".bz2")),
            ("foo.gz", (".gz", None, ".gz")),
            ("foo.zip", (".zip", ".zip", None)),
            ("foo.xz", (".xz", None, ".xz")),
            ("foo.bar.tar.gz", (".tar.gz", ".tar", ".gz")),
            ("foo.bar.gz", (".gz", None, ".gz")),
            ("foo.bar.zip", (".zip", ".zip", None)),
        ],
    )
    def test_detect_file_type(self, file, expected):
        assert utils._detect_file_type(file) == expected

    @pytest.mark.parametrize("file", ["foo", "foo.tar.baz", "foo.bar"])
    def test_detect_file_type_incompatible(self, file):
        # tests detect file type for no extension, unknown compression and unknown partial extension
        with pytest.raises(RuntimeError):
            utils._detect_file_type(file)

    @pytest.mark.parametrize("extension", [".bz2", ".gz", ".xz"])
    def test_decompress(self, extension, tmpdir):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}{extension}"
            compressed_file_opener = _COMPRESSED_FILE_OPENERS[extension]

            with compressed_file_opener(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        compressed, file, content = create_compressed(tmpdir)

        utils._decompress(compressed)

        assert os.path.exists(file)

        with open(file) as fh:
            assert fh.read() == content

    def test_decompress_no_compression(self):
        with pytest.raises(RuntimeError):
            utils._decompress("foo.tar")

    def test_decompress_remove_finished(self, tmpdir):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}.gz"

            with gzip.open(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        compressed, file, content = create_compressed(tmpdir)

        utils.extract_archive(compressed, tmpdir, remove_finished=True)

        assert not os.path.exists(compressed)

    @pytest.mark.parametrize("extension", [".gz", ".xz"])
    @pytest.mark.parametrize("remove_finished", [True, False])
    def test_extract_archive_defer_to_decompress(self, extension, remove_finished, mocker):
        filename = "foo"
        file = f"{filename}{extension}"

        mocked = mocker.patch("torchvision.datasets.utils._decompress")
        utils.extract_archive(file, remove_finished=remove_finished)

        mocked.assert_called_once_with(file, filename, remove_finished=remove_finished)

    def test_extract_zip(self, tmpdir):
        def create_archive(root, content="this is the content"):
            file = os.path.join(root, "dst.txt")
            archive = os.path.join(root, "archive.zip")

            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(os.path.basename(file), content)

            return archive, file, content

        archive, file, content = create_archive(tmpdir)

        utils.extract_archive(archive, tmpdir)

        assert os.path.exists(file)

        with open(file) as fh:
            assert fh.read() == content

    @pytest.mark.parametrize(
        "extension, mode", [(".tar", "w"), (".tar.gz", "w:gz"), (".tgz", "w:gz"), (".tar.xz", "w:xz")]
    )
    def test_extract_tar(self, extension, mode, tmpdir):
        def create_archive(root, extension, mode, content="this is the content"):
            src = os.path.join(root, "src.txt")
            dst = os.path.join(root, "dst.txt")
            archive = os.path.join(root, f"archive{extension}")

            with open(src, "w") as fh:
                fh.write(content)

            with tarfile.open(archive, mode=mode) as fh:
                fh.add(src, arcname=os.path.basename(dst))

            return archive, dst, content

        archive, file, content = create_archive(tmpdir, extension, mode)

        utils.extract_archive(archive, tmpdir)

        assert os.path.exists(file)

        with open(file) as fh:
            assert fh.read() == content

    def test_verify_str_arg(self):
        assert "a" == utils.verify_str_arg("a", "arg", ("a",))
        pytest.raises(ValueError, utils.verify_str_arg, 0, ("a",), "arg")
        pytest.raises(ValueError, utils.verify_str_arg, "b", ("a",), "arg")

    @pytest.mark.parametrize(
        ("dtype", "actual_hex", "expected_hex"),
        [
            (torch.uint8, "01 23 45 67 89 AB CD EF", "01 23 45 67 89 AB CD EF"),
            (torch.float16, "01 23 45 67 89 AB CD EF", "23 01 67 45 AB 89 EF CD"),
            (torch.int32, "01 23 45 67 89 AB CD EF", "67 45 23 01 EF CD AB 89"),
            (torch.float64, "01 23 45 67 89 AB CD EF", "EF CD AB 89 67 45 23 01"),
        ],
    )
    def test_flip_byte_order(self, dtype, actual_hex, expected_hex):
        def to_tensor(hex):
            return torch.frombuffer(bytes.fromhex(hex), dtype=dtype)

        assert_equal(
            utils._flip_byte_order(to_tensor(actual_hex)),
            to_tensor(expected_hex),
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_error_msg"),
    [
        (dict(is_valid_file=lambda path: pathlib.Path(path).suffix in {".png", ".jpeg"}), "classes c"),
        (dict(extensions=".png"), re.escape("classes b, c. Supported extensions are: .png")),
        (dict(extensions=(".png", ".jpeg")), re.escape("classes c. Supported extensions are: .png, .jpeg")),
    ],
)
def test_make_dataset_no_valid_files(tmpdir, kwargs, expected_error_msg):
    tmpdir = pathlib.Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "a" / "a.png").touch()

    (tmpdir / "b").mkdir()
    (tmpdir / "b" / "b.jpeg").touch()

    (tmpdir / "c").mkdir()
    (tmpdir / "c" / "c.unknown").touch()

    with pytest.raises(FileNotFoundError, match=expected_error_msg):
        make_dataset(str(tmpdir), **kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
