import bz2
import os
import torchvision.datasets.utils as utils
import pytest
import zipfile
import tarfile
import gzip
import warnings
from torch._utils_internal import get_file_path_2
from urllib.error import URLError
import itertools
import lzma

from common_utils import get_tmp_dir
from torchvision.datasets.utils import _COMPRESSED_FILE_OPENERS


TEST_FILE = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), 'assets', 'encode_jpeg', 'grace_hopper_517x606.jpg')


class TestDatasetsUtils:

    def test_check_md5(self):
        fpath = TEST_FILE
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        assert utils.check_md5(fpath, correct_md5)
        assert not utils.check_md5(fpath, false_md5)

    def test_check_integrity(self):
        existing_fpath = TEST_FILE
        nonexisting_fpath = ''
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        assert utils.check_integrity(existing_fpath, correct_md5)
        assert not utils.check_integrity(existing_fpath, false_md5)
        assert utils.check_integrity(existing_fpath)
        assert not utils.check_integrity(nonexisting_fpath)

    def test_get_google_drive_file_id(self):
        url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"
        expected = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"

        actual = utils._get_google_drive_file_id(url)
        assert actual == expected

    def test_get_google_drive_file_id_invalid_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

        assert utils._get_google_drive_file_id(url) is None

    @pytest.mark.parametrize('file, expected', [
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
        ("foo.bar.zip", (".zip", ".zip", None))])
    def test_detect_file_type(self, file, expected):
        assert utils._detect_file_type(file) == expected

    @pytest.mark.parametrize('file', ["foo", "foo.tar.baz", "foo.bar"])
    def test_detect_file_type_incompatible(self, file):
        # tests detect file type for no extension, unknown compression and unknown partial extension
        with pytest.raises(RuntimeError):
            utils._detect_file_type(file)

    @pytest.mark.parametrize('extension', [".bz2", ".gz", ".xz"])
    def test_decompress(self, extension):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}{extension}"
            compressed_file_opener = _COMPRESSED_FILE_OPENERS[extension]

            with compressed_file_opener(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        with get_tmp_dir() as temp_dir:
            compressed, file, content = create_compressed(temp_dir)

            utils._decompress(compressed)

            assert os.path.exists(file)

            with open(file, "r") as fh:
                assert fh.read() == content

    def test_decompress_no_compression(self):
        with pytest.raises(RuntimeError):
            utils._decompress("foo.tar")

    def test_decompress_remove_finished(self):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}.gz"

            with gzip.open(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        with get_tmp_dir() as temp_dir:
            compressed, file, content = create_compressed(temp_dir)

            utils.extract_archive(compressed, temp_dir, remove_finished=True)

            assert not os.path.exists(compressed)

    @pytest.mark.parametrize('extension', [".gz", ".xz"])
    @pytest.mark.parametrize('remove_finished', [True, False])
    def test_extract_archive_defer_to_decompress(self, extension, remove_finished, mocker):
        filename = "foo"
        file = f"{filename}{extension}"

        mocked = mocker.patch("torchvision.datasets.utils._decompress")
        utils.extract_archive(file, remove_finished=remove_finished)

        mocked.assert_called_once_with(file, filename, remove_finished=remove_finished)

    def test_extract_zip(self):
        def create_archive(root, content="this is the content"):
            file = os.path.join(root, "dst.txt")
            archive = os.path.join(root, "archive.zip")

            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(os.path.basename(file), content)

            return archive, file, content

        with get_tmp_dir() as temp_dir:
            archive, file, content = create_archive(temp_dir)

            utils.extract_archive(archive, temp_dir)

            assert os.path.exists(file)

            with open(file, "r") as fh:
                assert fh.read() == content

    @pytest.mark.parametrize('extension, mode', [
        ('.tar', 'w'), ('.tar.gz', 'w:gz'), ('.tgz', 'w:gz'), ('.tar.xz', 'w:xz')])
    def test_extract_tar(self, extension, mode):
        def create_archive(root, extension, mode, content="this is the content"):
            src = os.path.join(root, "src.txt")
            dst = os.path.join(root, "dst.txt")
            archive = os.path.join(root, f"archive{extension}")

            with open(src, "w") as fh:
                fh.write(content)

            with tarfile.open(archive, mode=mode) as fh:
                fh.add(src, arcname=os.path.basename(dst))

            return archive, dst, content

        with get_tmp_dir() as temp_dir:
            archive, file, content = create_archive(temp_dir, extension, mode)

            utils.extract_archive(archive, temp_dir)

            assert os.path.exists(file)

            with open(file, "r") as fh:
                assert fh.read() == content

    def test_verify_str_arg(self):
        assert "a" == utils.verify_str_arg("a", "arg", ("a",))
        pytest.raises(ValueError, utils.verify_str_arg, 0, ("a",), "arg")
        pytest.raises(ValueError, utils.verify_str_arg, "b", ("a",), "arg")


if __name__ == '__main__':
    pytest.main([__file__])
