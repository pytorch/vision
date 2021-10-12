"""This file should contain all tests that need access to the internet (apart
from the ones in test_datasets_download.py)

We want to bundle all internet-related tests in one file, so the file can be
cleanly ignored in FB internal test infra.
"""

import os
from urllib.error import URLError

import pytest
import torchvision.datasets.utils as utils


class TestDatasetUtils:
    def test_download_url(self, tmpdir):
        url = "http://github.com/pytorch/vision/archive/master.zip"
        try:
            utils.download_url(url, tmpdir)
            assert len(os.listdir(tmpdir)) != 0
        except URLError:
            pytest.skip(f"could not download test file '{url}'")

    def test_download_url_retry_http(self, tmpdir):
        url = "https://github.com/pytorch/vision/archive/master.zip"
        try:
            utils.download_url(url, tmpdir)
            assert len(os.listdir(tmpdir)) != 0
        except URLError:
            pytest.skip(f"could not download test file '{url}'")

    def test_download_url_dont_exist(self, tmpdir):
        url = "http://github.com/pytorch/vision/archive/this_doesnt_exist.zip"
        with pytest.raises(URLError):
            utils.download_url(url, tmpdir)

    def test_download_url_dispatch_download_from_google_drive(self, mocker, tmpdir):
        url = "https://drive.google.com/file/d/1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV/view"

        id = "1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV"
        filename = "filename"
        md5 = "md5"

        mocked = mocker.patch("torchvision.datasets.utils.download_file_from_google_drive")
        utils.download_url(url, tmpdir, filename, md5)

        mocked.assert_called_once_with(id, tmpdir, filename, md5)


if __name__ == "__main__":
    pytest.main([__file__])
