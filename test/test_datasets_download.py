import contextlib
import itertools
import time
import unittest.mock
from datetime import datetime
from os import path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import pytest

from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity

from common_utils import get_tmp_dir
from fakedata_generation import places365_root


def limit_requests_per_time(min_secs_between_requests=2.0):
    last_requests = {}

    def outer_wrapper(fn):
        def inner_wrapper(request, *args, **kwargs):
            url = request.full_url if isinstance(request, Request) else request

            netloc = urlparse(url).netloc
            last_request = last_requests.get(netloc)
            if last_request is not None:
                elapsed_secs = (datetime.now() - last_request).total_seconds()
                delta = min_secs_between_requests - elapsed_secs
                if delta > 0:
                    time.sleep(delta)

            response = fn(request, *args, **kwargs)
            last_requests[netloc] = datetime.now()

            return response

        return inner_wrapper

    return outer_wrapper


urlopen = limit_requests_per_time()(urlopen)


@contextlib.contextmanager
def log_download_attempts(patch=True):
    urls_and_md5s = set()
    with unittest.mock.patch("torchvision.datasets.utils.download_url", wraps=None if patch else download_url) as mock:
        try:
            yield urls_and_md5s
        finally:
            for args, kwargs in mock.call_args_list:
                url = args[0]
                md5 = args[-1] if len(args) == 4 else kwargs.get("md5")
                urls_and_md5s.add((url, md5))


def retry(fn, times=1, wait=5.0):
    msgs = []
    for _ in range(times + 1):
        try:
            return fn()
        except AssertionError as error:
            msgs.append(str(error))
            time.sleep(wait)
    else:
        raise AssertionError(
            "\n".join(
                (
                    f"Assertion failed {times + 1} times with {wait:.1f} seconds intermediate wait time.\n",
                    *(f"{idx}: {error}" for idx, error in enumerate(msgs, 1)),
                )
            )
        )


def assert_server_response_ok(response, url=None):
    msg = f"The server returned status code {response.code}"
    if url is not None:
        msg += f"for the the URL {url}"
    assert 200 <= response.code < 300, msg


def assert_url_is_accessible(url):
    request = Request(url, headers=dict(method="HEAD"))
    response = urlopen(request)
    assert_server_response_ok(response, url)


def assert_file_downloads_correctly(url, md5):
    with get_tmp_dir() as root:
        file = path.join(root, path.basename(url))
        with urlopen(url) as response, open(file, "wb") as fh:
            assert_server_response_ok(response, url)
            fh.write(response.read())

        assert check_integrity(file, md5=md5), "The MD5 checksums mismatch"


class DownloadConfig:
    def __init__(self, url, md5=None, id=None):
        self.url = url
        self.md5 = md5
        self.id = id or url


def make_parametrize_kwargs(download_configs):
    argvalues = []
    ids = []
    for config in download_configs:
        argvalues.append((config.url, config.md5))
        ids.append(config.id)

    return dict(argnames="url, md5", argvalues=argvalues, ids=ids)


def places365():
    with log_download_attempts(patch=False) as urls_and_md5s:
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with places365_root(split=split, small=small) as places365:
                root, data = places365

                datasets.Places365(root, split=split, small=small, download=True)

    return [DownloadConfig(url, md5=md5, id=f"Places365, {url}") for url, md5 in urls_and_md5s]


@pytest.mark.parametrize(**make_parametrize_kwargs(itertools.chain(places365(),)))
def test_url_is_accessible(url, md5):
    retry(lambda: assert_url_is_accessible(url))


@pytest.mark.parametrize(**make_parametrize_kwargs(itertools.chain()))
def test_file_downloads_correctly(url, md5):
    retry(lambda: assert_file_downloads_correctly(url, md5))
