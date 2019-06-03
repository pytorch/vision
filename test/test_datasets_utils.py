import os
import shutil
import tempfile
import torchvision.datasets.utils as utils
import unittest
import zipfile
import tarfile
import gzip

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'assets', 'grace_hopper_517x606.jpg')


class Tester(unittest.TestCase):

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

    def test_download_url(self):
        temp_dir = tempfile.mkdtemp()
        url = "http://github.com/pytorch/vision/archive/master.zip"
        utils.download_url(url, temp_dir)
        assert not len(os.listdir(temp_dir)) == 0, 'The downloaded root directory is empty after download.'
        shutil.rmtree(temp_dir)

    def test_download_url_retry_http(self):
        temp_dir = tempfile.mkdtemp()
        url = "https://github.com/pytorch/vision/archive/master.zip"
        utils.download_url(url, temp_dir)
        assert not len(os.listdir(temp_dir)) == 0, 'The downloaded root directory is empty after download.'
        shutil.rmtree(temp_dir)

    def test_extract_zip(self):
        temp_dir = tempfile.mkdtemp()
        with tempfile.NamedTemporaryFile(suffix='.zip') as f:
            with zipfile.ZipFile(f, 'w') as zf:
                zf.writestr('file.tst', 'this is the content')
            utils.extract_archive(f.name, temp_dir)
            assert os.path.exists(os.path.join(temp_dir, 'file.tst'))
            with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                data = nf.read()
            assert data == 'this is the content'
        shutil.rmtree(temp_dir)

    def test_extract_tar(self):
        for ext, mode in zip(['.tar', '.tar.gz'], ['w', 'w:gz']):
            temp_dir = tempfile.mkdtemp()
            with tempfile.NamedTemporaryFile() as bf:
                bf.write("this is the content".encode())
                bf.seek(0)
                with tempfile.NamedTemporaryFile(suffix=ext) as f:
                    with tarfile.open(f.name, mode=mode) as zf:
                        zf.add(bf.name, arcname='file.tst')
                    utils.extract_archive(f.name, temp_dir)
                    assert os.path.exists(os.path.join(temp_dir, 'file.tst'))
                    with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                        data = nf.read()
                    assert data == 'this is the content', data
            shutil.rmtree(temp_dir)

    def test_extract_gzip(self):
        temp_dir = tempfile.mkdtemp()
        with tempfile.NamedTemporaryFile(suffix='.gz') as f:
            with gzip.GzipFile(f.name, 'wb') as zf:
                zf.write('this is the content'.encode())
            utils.extract_archive(f.name, temp_dir)
            f_name = os.path.join(temp_dir, os.path.splitext(os.path.basename(f.name))[0])
            assert os.path.exists(f_name)
            with open(os.path.join(f_name), 'r') as nf:
                data = nf.read()
            assert data == 'this is the content', data
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
