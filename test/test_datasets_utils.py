import os
import torchvision.datasets.utils as utils
import unittest
import unittest.mock
import zipfile
import tarfile
import gzip
import warnings
from torch._utils_internal import get_file_path_2
from urllib.error import URLError
import itertools
import lzma

from common_utils import get_tmp_dir, call_args_to_kwargs_only


TEST_FILE = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), 'assets', 'encode_jpeg', 'grace_hopper_517x606.jpg')


class Tester(unittest.TestCase):

    def test_check_md5(self):
        fpath = TEST_FILE
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        self.assertTrue(utils.check_md5(fpath, correct_md5))
        self.assertFalse(utils.check_md5(fpath, false_md5))

    def test_check_integrity(self):
        existing_fpath = TEST_FILE
        nonexisting_fpath = ''
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        self.assertTrue(utils.check_integrity(existing_fpath, correct_md5))
        self.assertFalse(utils.check_integrity(existing_fpath, false_md5))
        self.assertTrue(utils.check_integrity(existing_fpath))
        self.assertFalse(utils.check_integrity(nonexisting_fpath))

    def test_get_redirect_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        expected = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        actual = utils._get_redirect_url(url)
        assert actual == expected

    def test_get_redirect_url_max_hops_exceeded(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        with self.assertRaises(RecursionError):
            utils._get_redirect_url(url, max_hops=0)

    def test_get_google_drive_file_id(self):
        url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"
        expected = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"

        actual = utils._get_google_drive_file_id(url)
        assert actual == expected

    def test_get_google_drive_file_id_invalid_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

        assert utils._get_google_drive_file_id(url) is None

    def test_download_url(self):
        with get_tmp_dir() as temp_dir:
            url = "http://github.com/pytorch/vision/archive/master.zip"
            try:
                utils.download_url(url, temp_dir)
                self.assertFalse(len(os.listdir(temp_dir)) == 0)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_download_url_retry_http(self):
        with get_tmp_dir() as temp_dir:
            url = "https://github.com/pytorch/vision/archive/master.zip"
            try:
                utils.download_url(url, temp_dir)
                self.assertFalse(len(os.listdir(temp_dir)) == 0)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_download_url_dont_exist(self):
        with get_tmp_dir() as temp_dir:
            url = "http://github.com/pytorch/vision/archive/this_doesnt_exist.zip"
            with self.assertRaises(URLError):
                utils.download_url(url, temp_dir)

    @unittest.mock.patch("torchvision.datasets.utils.download_file_from_google_drive")
    def test_download_url_dispatch_download_from_google_drive(self, mock):
        url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
        filename = "filename"
        md5 = "md5"

        with get_tmp_dir() as root:
            utils.download_url(url, root, filename, md5)

        mock.assert_called_once_with(id, root, filename, md5)

    def test_detect_file_type(self):
        for file, expected in [
            ("foo.tar.xz", (".tar.xz", ".tar", ".xz")),
            ("foo.tar", (".tar", ".tar", None)),
            ("foo.tar.gz", (".tar.gz", ".tar", ".gz")),
            ("foo.tgz", (".tgz", ".tar", ".gz")),
            ("foo.gz", (".gz", None, ".gz")),
            ("foo.zip", (".zip", ".zip", None)),
            ("foo.xz", (".xz", None, ".xz")),
        ]:
            with self.subTest(file=file):
                self.assertSequenceEqual(utils._detect_file_type(file), expected)

    def test_detect_file_type_no_ext(self):
        with self.assertRaises(RuntimeError):
            utils._detect_file_type("foo")

    def test_detect_file_type_to_many_exts(self):
        with self.assertRaises(RuntimeError):
            utils._detect_file_type("foo.bar.tar.gz")

    def test_detect_file_type_unknown_archive_type(self):
        with self.assertRaises(RuntimeError):
            utils._detect_file_type("foo.bar.gz")

    def test_detect_file_type_unknown_compression(self):
        with self.assertRaises(RuntimeError):
            utils._detect_file_type("foo.tar.baz")

    def test_detect_file_type_unknown_partial_ext(self):
        with self.assertRaises(RuntimeError):
            utils._detect_file_type("foo.bar")

    def test_decompress_gzip(self):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}.gz"

            with gzip.open(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        with get_tmp_dir() as temp_dir:
            compressed, file, content = create_compressed(temp_dir)

            utils._decompress(compressed)

            self.assertTrue(os.path.exists(file))

            with open(file, "r") as fh:
                self.assertEqual(fh.read(), content)

    def test_decompress_lzma(self):
        def create_compressed(root, content="this is the content"):
            file = os.path.join(root, "file")
            compressed = f"{file}.xz"

            with lzma.open(compressed, "wb") as fh:
                fh.write(content.encode())

            return compressed, file, content

        with get_tmp_dir() as temp_dir:
            compressed, file, content = create_compressed(temp_dir)

            utils.extract_archive(compressed, temp_dir)

            self.assertTrue(os.path.exists(file))

            with open(file, "r") as fh:
                self.assertEqual(fh.read(), content)

    def test_decompress_no_compression(self):
        with self.assertRaises(RuntimeError):
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

            self.assertFalse(os.path.exists(compressed))

    def test_extract_archive_defer_to_decompress(self):
        filename = "foo"
        for ext, remove_finished in itertools.product((".gz", ".xz"), (True, False)):
            with self.subTest(ext=ext, remove_finished=remove_finished):
                with unittest.mock.patch("torchvision.datasets.utils._decompress") as mock:
                    file = f"{filename}{ext}"
                    utils.extract_archive(file, remove_finished=remove_finished)

                mock.assert_called_once()
                self.assertEqual(
                    call_args_to_kwargs_only(mock.call_args, utils._decompress),
                    dict(from_path=file, to_path=filename, remove_finished=remove_finished),
                )

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

            self.assertTrue(os.path.exists(file))

            with open(file, "r") as fh:
                self.assertEqual(fh.read(), content)

    def test_extract_tar(self):
        def create_archive(root, ext, mode, content="this is the content"):
            src = os.path.join(root, "src.txt")
            dst = os.path.join(root, "dst.txt")
            archive = os.path.join(root, f"archive{ext}")

            with open(src, "w") as fh:
                fh.write(content)

            with tarfile.open(archive, mode=mode) as fh:
                fh.add(src, arcname=os.path.basename(dst))

            return archive, dst, content

        for ext, mode in zip(['.tar', '.tar.gz', '.tgz'], ['w', 'w:gz', 'w:gz']):
            with get_tmp_dir() as temp_dir:
                archive, file, content = create_archive(temp_dir, ext, mode)

                utils.extract_archive(archive, temp_dir)

                self.assertTrue(os.path.exists(file))

                with open(file, "r") as fh:
                    self.assertEqual(fh.read(), content)

    def test_extract_tar_xz(self):
        def create_archive(root, ext, mode, content="this is the content"):
            src = os.path.join(root, "src.txt")
            dst = os.path.join(root, "dst.txt")
            archive = os.path.join(root, f"archive{ext}")

            with open(src, "w") as fh:
                fh.write(content)

            with tarfile.open(archive, mode=mode) as fh:
                fh.add(src, arcname=os.path.basename(dst))

            return archive, dst, content

        for ext, mode in zip(['.tar.xz'], ['w:xz']):
            with get_tmp_dir() as temp_dir:
                archive, file, content = create_archive(temp_dir, ext, mode)

                utils.extract_archive(archive, temp_dir)

                self.assertTrue(os.path.exists(file))

                with open(file, "r") as fh:
                    self.assertEqual(fh.read(), content)

    def test_verify_str_arg(self):
        self.assertEqual("a", utils.verify_str_arg("a", "arg", ("a",)))
        self.assertRaises(ValueError, utils.verify_str_arg, 0, ("a",), "arg")
        self.assertRaises(ValueError, utils.verify_str_arg, "b", ("a",), "arg")


if __name__ == '__main__':
    unittest.main()
