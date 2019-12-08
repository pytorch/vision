import os
import sys
import tempfile
import torchvision.datasets.utils as utils
import unittest
import zipfile
import tarfile
import gzip
import warnings
from torch._six import PY2
from torch._utils_internal import get_file_path_2

from common_utils import get_tmp_dir

if sys.version_info < (3,):
    from urllib2 import URLError
else:
    from urllib.error import URLError


TEST_FILE = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), 'assets', 'grace_hopper_517x606.jpg')


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

    @unittest.skipIf(PY2, "https://github.com/pytorch/vision/issues/1268")
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

    @unittest.skipIf(PY2, "https://github.com/pytorch/vision/issues/1268")
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

    @unittest.skipIf(sys.version_info < (3,), "Python2 doesn't raise error")
    def test_download_url_dont_exist(self):
        with get_tmp_dir() as temp_dir:
            url = "http://github.com/pytorch/vision/archive/this_doesnt_exist.zip"
            with self.assertRaises(URLError):
                utils.download_url(url, temp_dir)

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_extract_zip(self):
        with get_tmp_dir() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix='.zip') as f:
                with zipfile.ZipFile(f, 'w') as zf:
                    zf.writestr('file.tst', 'this is the content')
                utils.extract_archive(f.name, temp_dir)
                self.assertTrue(os.path.exists(os.path.join(temp_dir, 'file.tst')))
                with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                    data = nf.read()
                self.assertEqual(data, 'this is the content')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_extract_tar(self):
        for ext, mode in zip(['.tar', '.tar.gz', '.tgz'], ['w', 'w:gz', 'w:gz']):
            with get_tmp_dir() as temp_dir:
                with tempfile.NamedTemporaryFile() as bf:
                    bf.write("this is the content".encode())
                    bf.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=ext) as f:
                        with tarfile.open(f.name, mode=mode) as zf:
                            zf.add(bf.name, arcname='file.tst')
                        utils.extract_archive(f.name, temp_dir)
                        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'file.tst')))
                        with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                            data = nf.read()
                        self.assertEqual(data, 'this is the content')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    @unittest.skipIf(sys.version_info < (3,), "Extracting .tar.xz files is not supported under Python 2.x")
    def test_extract_tar_xz(self):
        for ext, mode in zip(['.tar.xz'], ['w:xz']):
            with get_tmp_dir() as temp_dir:
                with tempfile.NamedTemporaryFile() as bf:
                    bf.write("this is the content".encode())
                    bf.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=ext) as f:
                        with tarfile.open(f.name, mode=mode) as zf:
                            zf.add(bf.name, arcname='file.tst')
                        utils.extract_archive(f.name, temp_dir)
                        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'file.tst')))
                        with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                            data = nf.read()
                        self.assertEqual(data, 'this is the content')

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_extract_gzip(self):
        with get_tmp_dir() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix='.gz') as f:
                with gzip.GzipFile(f.name, 'wb') as zf:
                    zf.write('this is the content'.encode())
                utils.extract_archive(f.name, temp_dir)
                f_name = os.path.join(temp_dir, os.path.splitext(os.path.basename(f.name))[0])
                self.assertTrue(os.path.exists(f_name))
                with open(os.path.join(f_name), 'r') as nf:
                    data = nf.read()
                self.assertEqual(data, 'this is the content')

    def test_verify_str_arg(self):
        self.assertEqual("a", utils.verify_str_arg("a", "arg", ("a",)))
        self.assertRaises(ValueError, utils.verify_str_arg, 0, ("a",), "arg")
        self.assertRaises(ValueError, utils.verify_str_arg, "b", ("a",), "arg")


if __name__ == '__main__':
    unittest.main()
