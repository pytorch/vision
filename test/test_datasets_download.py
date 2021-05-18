import contextlib
import itertools
import time
import unittest.mock
from datetime import datetime
from distutils import dir_util
from os import path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import tempfile
import warnings

import pytest

from torchvision import datasets
from torchvision.datasets.utils import (
    download_url,
    check_integrity,
    download_file_from_google_drive,
    _get_redirect_url,
    USER_AGENT,
)

from common_utils import get_tmp_dir


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


def resolve_redirects(max_hops=3):
    def outer_wrapper(fn):
        def inner_wrapper(request, *args, **kwargs):
            initial_url = request.full_url if isinstance(request, Request) else request
            url = _get_redirect_url(initial_url, max_hops=max_hops)

            if url == initial_url:
                return fn(request, *args, **kwargs)

            warnings.warn(f"The URL {initial_url} ultimately redirects to {url}.")

            if not isinstance(request, Request):
                return fn(url, *args, **kwargs)

            request_attrs = {
                attr: getattr(request, attr) for attr in ("data", "headers", "origin_req_host", "unverifiable")
            }
            # the 'method' attribute does only exist if the request was created with it
            if hasattr(request, "method"):
                request_attrs["method"] = request.method

            return fn(Request(url, **request_attrs), *args, **kwargs)

        return inner_wrapper

    return outer_wrapper


urlopen = resolve_redirects()(urlopen)


@contextlib.contextmanager
def log_download_attempts(
    urls_and_md5s=None,
    file="utils",
    patch=True,
    mock_auxiliaries=None,
):
    def add_mock(stack, name, file, **kwargs):
        try:
            return stack.enter_context(unittest.mock.patch(f"torchvision.datasets.{file}.{name}", **kwargs))
        except AttributeError as error:
            if file != "utils":
                return add_mock(stack, name, "utils", **kwargs)
            else:
                raise pytest.UsageError from error

    if urls_and_md5s is None:
        urls_and_md5s = set()
    if mock_auxiliaries is None:
        mock_auxiliaries = patch

    with contextlib.ExitStack() as stack:
        url_mock = add_mock(stack, "download_url", file, wraps=None if patch else download_url)
        google_drive_mock = add_mock(
            stack, "download_file_from_google_drive", file, wraps=None if patch else download_file_from_google_drive
        )

        if mock_auxiliaries:
            add_mock(stack, "extract_archive", file)

        try:
            yield urls_and_md5s
        finally:
            for args, kwargs in url_mock.call_args_list:
                url = args[0]
                md5 = args[-1] if len(args) == 4 else kwargs.get("md5")
                urls_and_md5s.add((url, md5))

            for args, kwargs in google_drive_mock.call_args_list:
                id = args[0]
                url = f"https://drive.google.com/file/d/{id}"
                md5 = args[3] if len(args) == 4 else kwargs.get("md5")
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
    except URLError as error:
        raise AssertionError("The request timed out.") from error
    except HTTPError as error:
        raise AssertionError(f"The server returned {error.code}: {error.reason}.") from error
    except RecursionError as error:
        raise AssertionError(str(error)) from error


def assert_url_is_accessible(url, timeout=5.0):
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with assert_server_response_ok():
        urlopen(request, timeout=timeout)


def assert_file_downloads_correctly(url, md5, timeout=5.0):
    with get_tmp_dir() as root:
        file = path.join(root, path.basename(url))
        with assert_server_response_ok():
            with open(file, "wb") as fh:
                request = Request(url, headers={"User-Agent": USER_AGENT})
                response = urlopen(request, timeout=timeout)
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


def collect_download_configs(dataset_loader, name=None, **kwargs):
    urls_and_md5s = set()
    try:
        with log_download_attempts(urls_and_md5s=urls_and_md5s, **kwargs):
            dataset = dataset_loader()
    except Exception:
        dataset = None

    if name is None and dataset is not None:
        name = type(dataset).__name__

    return make_download_configs(urls_and_md5s, name)


# This is a workaround since fixtures, such as the built-in tmp_dir, can only be used within a test but not within a
# parametrization. Thus, we use a single root directory for all datasets and remove it when all download tests are run.
ROOT = tempfile.mkdtemp()


@pytest.fixture(scope="module", autouse=True)
def root():
    yield ROOT
    dir_util.remove_tree(ROOT)


def places365():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.Places365(ROOT, split=split, small=small, download=True),
                name=f"Places365, {split}, {'small' if small else 'large'}",
                file="places365",
            )
            for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True))
        ]
    )


def caltech101():
    return collect_download_configs(lambda: datasets.Caltech101(ROOT, download=True), name="Caltech101")


def caltech256():
    return collect_download_configs(lambda: datasets.Caltech256(ROOT, download=True), name="Caltech256")


def cifar10():
    return collect_download_configs(lambda: datasets.CIFAR10(ROOT, download=True), name="CIFAR10")


def cifar100():
    return collect_download_configs(lambda: datasets.CIFAR100(ROOT, download=True), name="CIFAR100")


def voc():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.VOCSegmentation(ROOT, year=year, download=True),
                name=f"VOC, {year}",
                file="voc",
            )
            for year in ("2007", "2007-test", "2008", "2009", "2010", "2011", "2012")
        ]
    )


def mnist():
    with unittest.mock.patch.object(datasets.MNIST, "mirrors", datasets.MNIST.mirrors[-1:]):
        return collect_download_configs(lambda: datasets.MNIST(ROOT, download=True), name="MNIST")


def fashion_mnist():
    return collect_download_configs(lambda: datasets.FashionMNIST(ROOT, download=True), name="FashionMNIST")


def kmnist():
    return collect_download_configs(lambda: datasets.KMNIST(ROOT, download=True), name="KMNIST")


def emnist():
    # the 'split' argument can be any valid one, since everything is downloaded anyway
    return collect_download_configs(lambda: datasets.EMNIST(ROOT, split="byclass", download=True), name="EMNIST")


def qmnist():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.QMNIST(ROOT, what=what, download=True),
                name=f"QMNIST, {what}",
                file="mnist",
            )
            for what in ("train", "test", "nist")
        ]
    )


def omniglot():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.Omniglot(ROOT, background=background, download=True),
                name=f"Omniglot, {'background' if background else 'evaluation'}",
            )
            for background in (True, False)
        ]
    )


def phototour():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.PhotoTour(ROOT, name=name, download=True),
                name=f"PhotoTour, {name}",
                file="phototour",
            )
            # The names postfixed with '_harris' point to the domain 'matthewalunbrown.com'. For some reason all
            # requests timeout from within CI. They are disabled until this is resolved.
            for name in ("notredame", "yosemite", "liberty")  # "notredame_harris", "yosemite_harris", "liberty_harris"
        ]
    )


def sbdataset():
    return collect_download_configs(
        lambda: datasets.SBDataset(ROOT, download=True),
        name="SBDataset",
        file="voc",
    )


def sbu():
    return collect_download_configs(
        lambda: datasets.SBU(ROOT, download=True),
        name="SBU",
        file="sbu",
    )


def semeion():
    return collect_download_configs(
        lambda: datasets.SEMEION(ROOT, download=True),
        name="SEMEION",
        file="semeion",
    )


def stl10():
    return collect_download_configs(
        lambda: datasets.STL10(ROOT, download=True),
        name="STL10",
    )


def svhn():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.SVHN(ROOT, split=split, download=True),
                name=f"SVHN, {split}",
                file="svhn",
            )
            for split in ("train", "test", "extra")
        ]
    )


def usps():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda: datasets.USPS(ROOT, train=train, download=True),
                name=f"USPS, {'train' if train else 'test'}",
                file="usps",
            )
            for train in (True, False)
        ]
    )


def celeba():
    return collect_download_configs(
        lambda: datasets.CelebA(ROOT, download=True),
        name="CelebA",
        file="celeba",
    )


def widerface():
    return collect_download_configs(
        lambda: datasets.WIDERFace(ROOT, download=True),
        name="WIDERFace",
        file="widerface",
    )


def kitti():
    return itertools.chain(
        *[
            collect_download_configs(
                lambda train=train: datasets.Kitti(ROOT, train=train, download=True),
                name=f"Kitti, {'train' if train else 'test'}",
                file="kitti",
            )
            for train in (True, False)
        ]
    )


def make_parametrize_kwargs(download_configs):
    argvalues = []
    ids = []
    for config in download_configs:
        argvalues.append((config.url, config.md5))
        ids.append(config.id)

    return dict(argnames=("url", "md5"), argvalues=argvalues, ids=ids)


@pytest.mark.parametrize(
    **make_parametrize_kwargs(
        itertools.chain(
            places365(),
            caltech101(),
            caltech256(),
            cifar10(),
            cifar100(),
            # The VOC download server is unstable. See https://github.com/pytorch/vision/issues/2953 for details.
            # voc(),
            mnist(),
            fashion_mnist(),
            kmnist(),
            emnist(),
            qmnist(),
            omniglot(),
            phototour(),
            sbdataset(),
            sbu(),
            semeion(),
            stl10(),
            svhn(),
            usps(),
            celeba(),
            widerface(),
            kitti(),
        )
    )
)
def test_url_is_accessible(url, md5):
    retry(lambda: assert_url_is_accessible(url))


@pytest.mark.parametrize(**make_parametrize_kwargs(itertools.chain()))
def test_file_downloads_correctly(url, md5):
    retry(lambda: assert_file_downloads_correctly(url, md5))
