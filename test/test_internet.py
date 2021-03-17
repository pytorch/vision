"""This file should contain all tests that need access to the internet (apart
from the ones in test_datasets_download.py)

We want to bundle all internet-related tests in one file, so the file can be
cleanly ignored in FB internal test infra.
"""

import os
import unittest
import unittest.mock
import warnings
from urllib.error import URLError

import torchvision.datasets.utils as utils
from common_utils import get_tmp_dir


class DatasetUtilsTester(unittest.TestCase):

    def test_get_redirect_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        expected = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        actual = utils._get_redirect_url(url)
        assert actual == expected

    def test_get_redirect_url_max_hops_exceeded(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        with self.assertRaises(RecursionError):
            utils._get_redirect_url(url, max_hops=0)

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
