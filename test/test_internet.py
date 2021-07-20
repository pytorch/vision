"""This file should contain all tests that need access to the internet (apart
from the ones in test_datasets_download.py)

We want to bundle all internet-related tests in one file, so the file can be
cleanly ignored in FB internal test infra.
"""

import os
import pytest
import warnings
from urllib.error import URLError
from urllib.request import Request

import torchvision.datasets.utils as utils
from common_utils import get_tmp_dir


@pytest.fixture
def patch_url_redirection(mocker):
    class Response:
        def __init__(self, url):
            self.url = url

    def factory(*urls):
        class PatchedOpener:
            def __init__(self, request_or_url, *args, **kwargs):
                self._request_or_url = request_or_url

            def __enter__(self):
                url = (
                    self._request_or_url.full_url
                    if isinstance(self._request_or_url.full_url, Request)
                    else self._request_or_url.full_url
                )

                if url == urls[-1]:
                    redirect_url = url
                else:
                    redirect_url = urls[urls.index(url) + 1]

                return Response(redirect_url)

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        mocker.patch("torchvision.datasets.utils.urllib.request.urlopen", new=PatchedOpener)

    return factory


class TestDatasetUtils:
    def test_get_redirect_url(self, patch_url_redirection):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        expected_redirected_url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        patch_url_redirection(url, expected_redirected_url)

        actual = utils._get_redirect_url(url)
        assert actual == expected_redirected_url

    def test_get_redirect_url_max_hops_exceeded(self, patch_url_redirection):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        redirected_url = "https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"

        patch_url_redirection(url, redirected_url)

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
