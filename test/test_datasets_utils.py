import os
import shutil
import tempfile
import torchvision.datasets.utils as utils
import unittest
import zipfile

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
        
    def test_ziplookup(self):
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, "ziplookup.zip")
        try:
            z = zipfile.ZipFile(temp_filename, "w", zipfile.ZIP_STORED, allowZip64=True)
            z.write(TEST_FILE, "hopper.jpg")
            z.write(TEST_FILE)
            z.write(TEST_FILE, "hopper79.jpg")
            z.write(TEST_FILE, "somepath/hopper.jpg")
            z.close()
            
            lookup = utils.ForkSafeZipLookup(temp_filename)
            f = lookup["hopper.jpg"]
            assert f.name.endswith(".jpg")
            f = lookup["somepath/hopper.jpg"]
            assert f.name.endswith(".jpg")
            try:
                f = lookup["does_not_exist.jpg"]
                assert False, "Should not return something for non-existant file"
            except KeyError:
                pass
            del lookup
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
