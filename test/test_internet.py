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
    @pytest.mark.parametrize("temp_dir", ("tmpdir", "tmp_path"))
    def test_download_url(self, temp_dir, request):
        temp_dir = request.getfixturevalue(temp_dir)
        url = "http://github.com/pytorch/vision/archive/master.zip"
        try:
            utils.download_url(url, temp_dir)
            assert len(os.listdir(temp_dir)) != 0
        except URLError:
            pytest.skip(f"could not download test file '{url}'")

    @pytest.mark.parametrize("temp_dir", ("tmpdir", "tmp_path"))
    def test_download_url_retry_http(self, temp_dir, request):
        temp_dir = request.getfixturevalue(temp_dir)
        url = "https://github.com/pytorch/vision/archive/master.zip"
        try:
            utils.download_url(url, temp_dir)
            assert len(os.listdir(temp_dir)) != 0
        except URLError:
            pytest.skip(f"could not download test file '{url}'")

    @pytest.mark.parametrize("temp_dir", ("tmpdir", "tmp_path"))
    def test_download_url_dont_exist(self, temp_dir, request):
        temp_dir = request.getfixturevalue(temp_dir)
        url = "http://github.com/pytorch/vision/archive/this_doesnt_exist.zip"
        with pytest.raises(URLError):
            utils.download_url(url, temp_dir)

    @pytest.mark.parametrize("temp_dir", ("tmpdir", "tmp_path"))
    def test_download_url_dispatch_download_from_google_drive(self, mocker, temp_dir, request):
        temp_dir = request.getfixturevalue(temp_dir)
        url = "https://drive.google.com/file/d/1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV/view"

        id = "1GO-BHUYRuvzr1Gtp2_fqXRsr9TIeYbhV"
        filename = "filename"
        md5 = "md5"

        mocked = mocker.patch("torchvision.datasets.utils.download_file_from_google_drive")
        utils.download_url(url, temp_dir, filename, md5)

        mocked.assert_called_once_with(id, os.path.expanduser(temp_dir), filename, md5)


if __name__ == "__main__":
    pytest.main([__file__])
