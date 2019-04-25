import os
import shutil
import tempfile
import torchvision.datasets.utils as utils
import unittest

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


if __name__ == '__main__':
    unittest.main()
