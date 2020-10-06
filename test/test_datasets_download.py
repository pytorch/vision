import contextlib
import itertools
import time
import unittest.mock
from datetime import datetime
from os import path
from urllib.error import HTTPError
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
def log_download_attempts(urls_and_md5s=None, patch=True, patch_auxiliaries=None):
    if urls_and_md5s is None:
        urls_and_md5s = set()
    if patch_auxiliaries is None:
        patch_auxiliaries = patch

    with contextlib.ExitStack() as stack:
        download_url_mock = stack.enter_context(
            unittest.mock.patch("torchvision.datasets.utils.download_url", wraps=None if patch else download_url)
        )
        if patch_auxiliaries:
            # download_and_extract_archive
            stack.enter_context(unittest.mock.patch("torchvision.datasets.utils.extract_archive"))
        try:
            yield urls_and_md5s
        finally:
            for args, kwargs in download_url_mock.call_args_list:
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


@contextlib.contextmanager
def assert_server_response_ok():
    try:
        yield
    except HTTPError as error:
        raise AssertionError(f"The server returned {error.code}: {error.reason}.") from error


def assert_url_is_accessible(url):
    request = Request(url, headers=dict(method="HEAD"))
    with assert_server_response_ok():
        urlopen(request)


def assert_file_downloads_correctly(url, md5):
    with get_tmp_dir() as root:
        file = path.join(root, path.basename(url))
        with assert_server_response_ok():
            with urlopen(url) as response, open(file, "wb") as fh:
                fh.write(response.read())

        assert check_integrity(file, md5=md5), "The MD5 checksums mismatch"


class DownloadConfig:
    def __init__(self, url, md5=None, id=None):
        self.url = url
        self.md5 = md5
        self.id = id or url

    def __repr__(self):
        return self.id


def make_download_configs(urls_and_md5s, name=None):
    return [
        DownloadConfig(url, md5=md5, id=f"{name}, {url}" if name is not None else None) for url, md5 in urls_and_md5s
    ]


def collect_download_configs(dataset_loader, name):
    try:
        with log_download_attempts() as urls_and_md5s:
            dataset_loader()
    except Exception:
        pass

    return make_download_configs(urls_and_md5s, name)


def places365():
    with log_download_attempts(patch=False) as urls_and_md5s:
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with places365_root(split=split, small=small) as places365:
                root, data = places365

                datasets.Places365(root, split=split, small=small, download=True)

    return make_download_configs(urls_and_md5s, "Places365")


def caltech101():
    return collect_download_configs(lambda: datasets.Caltech101(".", download=True), "Caltech101")


def caltech256():
    return collect_download_configs(lambda: datasets.Caltech256(".", download=True), "Caltech256")


def cifar10():
    return collect_download_configs(lambda: datasets.CIFAR10(".", download=True), "CIFAR10")


def cifar100():
    return collect_download_configs(lambda: datasets.CIFAR10(".", download=True), "CIFAR100")


def make_parametrize_kwargs(download_configs):
    argvalues = []
    ids = []
    for config in download_configs:
        argvalues.append((config.url, config.md5))
        ids.append(config.id)

    return dict(argnames=("url", "md5"), argvalues=argvalues, ids=ids)


@pytest.mark.parametrize(
    **make_parametrize_kwargs(itertools.chain(places365(), caltech101(), caltech256(), cifar10(), cifar100()))
)
def test_url_is_accessible(url, md5):
    retry(lambda: assert_url_is_accessible(url))


@pytest.mark.parametrize(**make_parametrize_kwargs(itertools.chain()))
def test_file_downloads_correctly(url, md5):
    retry(lambda: assert_file_downloads_correctly(url, md5))
