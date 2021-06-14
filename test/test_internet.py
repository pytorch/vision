"""This file should contain all tests that need access to the internet (apart
from the ones in test_datasets_download.py)

We want to bundle all internet-related tests in one file, so the file can be
cleanly ignored in FB internal test infra.
"""

import os
import pytest
import warnings
from urllib.error import URLError

import torchvision.datasets.utils as utils
from common_utils import get_tmp_dir


class TestDatasetUtils:

    def test_get_redirect_url(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        expected = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        actual = utils._get_redirect_url(url)
        assert actual == expected

    def test_get_redirect_url_max_hops_exceeded(self):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        with pytest.raises(RecursionError):
            utils._get_redirect_url(url, max_hops=0)

    def test_download_url(self):
        with get_tmp_dir() as temp_dir:
            url = "http://github.com/pytorch/vision/archive/master.zip"
            try:
                utils.download_url(url, temp_dir)
                assert len(os.listdir(temp_dir)) != 0
            except URLError:
                pytest.skip(f"could not download test file '{url}'")

    def test_download_url_retry_http(self):
        with get_tmp_dir() as temp_dir:
            url = "https://github.com/pytorch/vision/archive/master.zip"
            try:
                utils.download_url(url, temp_dir)
                assert len(os.listdir(temp_dir)) != 0
            except URLError:
                pytest.skip(f"could not download test file '{url}'")

    def test_download_url_dont_exist(self):
        with get_tmp_dir() as temp_dir:
            url = "http://github.com/pytorch/vision/archive/this_doesnt_exist.zip"
            with pytest.raises(URLError):
                utils.download_url(url, temp_dir)

    def test_download_url_dispatch_download_from_google_drive(self, mocker):
        url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
        filename = "filename"
        md5 = "md5"

        mocked = mocker.patch('torchvision.datasets.utils.download_file_from_google_drive')
        with get_tmp_dir() as root:
            utils.download_url(url, root, filename, md5)

        mocked.assert_called_once_with(id, root, filename, md5)


if __name__ == '__main__':
    pytest.main([__file__])
