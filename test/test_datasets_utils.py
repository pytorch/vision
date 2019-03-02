import shutil
import tempfile
import torch
import torchvision.datasets.utils as utils
import unittest


class Tester(unittest.TestCase):

    def test_download_url(self):
        temp_dir = tempfile.mkdtemp()
        url = "http://github.com/pytorch/vision/archive/master.zip"
        utils.download_url(url, temp_dir)
        shutil.rmtree(temp_dir)

    def test_download_url_retry_http(self):
        temp_dir = tempfile.mkdtemp()
        url = "https://github.com/pytorch/vision/archive/master.zip"
        utils.download_url(url, temp_dir)
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
