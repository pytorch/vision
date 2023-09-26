import contextlib
import itertools
import tempfile
import time
import traceback
import unittest.mock
import warnings
from datetime import datetime
from distutils import dir_util
from os import path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest
from torchvision import datasets
from torchvision.datasets.utils import _get_redirect_url, USER_AGENT


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
    urls,
    *,
    dataset_module,
):
    def maybe_add_mock(*, module, name, stack, lst=None):
        patcher = unittest.mock.patch(f"torchvision.datasets.{module}.{name}")

        try:
            mock = stack.enter_context(patcher)
        except AttributeError:
            return

        if lst is not None:
            lst.append(mock)

    with contextlib.ExitStack() as stack:
        download_url_mocks = []
        download_file_from_google_drive_mocks = []
        for module in [dataset_module, "utils"]:
            maybe_add_mock(module=module, name="download_url", stack=stack, lst=download_url_mocks)
            maybe_add_mock(
                module=module,
                name="download_file_from_google_drive",
                stack=stack,
                lst=download_file_from_google_drive_mocks,
            )
            maybe_add_mock(module=module, name="extract_archive", stack=stack)

        try:
            yield
        finally:
            for download_url_mock in download_url_mocks:
                for args, kwargs in download_url_mock.call_args_list:
                    urls.append(args[0] if args else kwargs["url"])

            for download_file_from_google_drive_mock in download_file_from_google_drive_mocks:
                for args, kwargs in download_file_from_google_drive_mock.call_args_list:
                    file_id = args[0] if args else kwargs["file_id"]
                    urls.append(f"https://drive.google.com/file/d/{file_id}")


def retry(fn, times=1, wait=5.0):
    tbs = []
    for _ in range(times + 1):
        try:
            return fn()
        except AssertionError as error:
            tbs.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
            time.sleep(wait)
    else:
        raise AssertionError(
            "\n".join(
                (
                    "\n",
                    *[f"{'_' * 40}  {idx:2d}  {'_' * 40}\n\n{tb}" for idx, tb in enumerate(tbs, 1)],
                    (
                        f"Assertion failed {times + 1} times with {wait:.1f} seconds intermediate wait time. "
                        f"You can find the the full tracebacks above."
                    ),
                )
            )
        )


@contextlib.contextmanager
def assert_server_response_ok():
    try:
        yield
    except HTTPError as error:
        raise AssertionError(f"The server returned {error.code}: {error.reason}.") from error
    except URLError as error:
        raise AssertionError(
            "Connection not possible due to SSL." if "SSL" in str(error) else "The request timed out."
        ) from error
    except RecursionError as error:
        raise AssertionError(str(error)) from error


def assert_url_is_accessible(url, timeout=5.0):
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with assert_server_response_ok():
        urlopen(request, timeout=timeout)


def collect_urls(dataset_cls, *args, **kwargs):
    urls = []
    with contextlib.suppress(Exception), log_download_attempts(
        urls, dataset_module=dataset_cls.__module__.split(".")[-1]
    ):
        dataset_cls(*args, **kwargs)

    return [(url, f"{dataset_cls.__name__}, {url}") for url in urls]


# This is a workaround since fixtures, such as the built-in tmp_dir, can only be used within a test but not within a
# parametrization. Thus, we use a single root directory for all datasets and remove it when all download tests are run.
ROOT = tempfile.mkdtemp()


@pytest.fixture(scope="module", autouse=True)
def root():
    yield ROOT
    dir_util.remove_tree(ROOT)


def places365():
    return itertools.chain.from_iterable(
        [
            collect_urls(
                datasets.Places365,
                ROOT,
                split=split,
                small=small,
                download=True,
            )
            for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True))
        ]
    )


def caltech101():
    return collect_urls(datasets.Caltech101, ROOT, download=True)


def caltech256():
    return collect_urls(datasets.Caltech256, ROOT, download=True)


def cifar10():
    return collect_urls(datasets.CIFAR10, ROOT, download=True)


def cifar100():
    return collect_urls(datasets.CIFAR100, ROOT, download=True)


def voc():
    # TODO: Also test the "2007-test" key
    return itertools.chain.from_iterable(
        [
            collect_urls(datasets.VOCSegmentation, ROOT, year=year, download=True)
            for year in ("2007", "2008", "2009", "2010", "2011", "2012")
        ]
    )


def mnist():
    with unittest.mock.patch.object(datasets.MNIST, "mirrors", datasets.MNIST.mirrors[-1:]):
        return collect_urls(datasets.MNIST, ROOT, download=True)


def fashion_mnist():
    return collect_urls(datasets.FashionMNIST, ROOT, download=True)


def kmnist():
    return collect_urls(datasets.KMNIST, ROOT, download=True)


def emnist():
    # the 'split' argument can be any valid one, since everything is downloaded anyway
    return collect_urls(datasets.EMNIST, ROOT, split="byclass", download=True)


def qmnist():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.QMNIST, ROOT, what=what, download=True) for what in ("train", "test", "nist")]
    )


def moving_mnist():
    return collect_urls(datasets.MovingMNIST, ROOT, download=True)


def omniglot():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.Omniglot, ROOT, background=background, download=True) for background in (True, False)]
    )


def phototour():
    return itertools.chain.from_iterable(
        [
            collect_urls(datasets.PhotoTour, ROOT, name=name, download=True)
            # The names postfixed with '_harris' point to the domain 'matthewalunbrown.com'. For some reason all
            # requests timeout from within CI. They are disabled until this is resolved.
            for name in ("notredame", "yosemite", "liberty")  # "notredame_harris", "yosemite_harris", "liberty_harris"
        ]
    )


def sbdataset():
    return collect_urls(datasets.SBDataset, ROOT, download=True)


def sbu():
    return collect_urls(datasets.SBU, ROOT, download=True)


def semeion():
    return collect_urls(datasets.SEMEION, ROOT, download=True)


def stl10():
    return collect_urls(datasets.STL10, ROOT, download=True)


def svhn():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.SVHN, ROOT, split=split, download=True) for split in ("train", "test", "extra")]
    )


def usps():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.USPS, ROOT, train=train, download=True) for train in (True, False)]
    )


def celeba():
    return collect_urls(datasets.CelebA, ROOT, download=True)


def widerface():
    return collect_urls(datasets.WIDERFace, ROOT, download=True)


def kinetics():
    return itertools.chain.from_iterable(
        [
            collect_urls(
                datasets.Kinetics,
                path.join(ROOT, f"Kinetics{num_classes}"),
                frames_per_clip=1,
                num_classes=num_classes,
                split=split,
                download=True,
            )
            for num_classes, split in itertools.product(("400", "600", "700"), ("train", "val"))
        ]
    )


def kitti():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.Kitti, ROOT, train=train, download=True) for train in (True, False)]
    )


def stanford_cars():
    return itertools.chain.from_iterable(
        [collect_urls(datasets.StanfordCars, ROOT, split=split, download=True) for split in ["train", "test"]]
    )


def url_parametrization(*dataset_urls_and_ids_fns):
    return pytest.mark.parametrize(
        "url",
        [
            pytest.param(url, id=id)
            for dataset_urls_and_ids_fn in dataset_urls_and_ids_fns
            for url, id in sorted(set(dataset_urls_and_ids_fn()))
        ],
    )


@url_parametrization(
    caltech101,
    caltech256,
    cifar10,
    cifar100,
    # The VOC download server is unstable. See https://github.com/pytorch/vision/issues/2953 for details.
    # voc,
    mnist,
    fashion_mnist,
    kmnist,
    emnist,
    qmnist,
    omniglot,
    phototour,
    sbdataset,
    semeion,
    stl10,
    svhn,
    usps,
    celeba,
    widerface,
    kinetics,
    kitti,
    places365,
    sbu,
)
def test_url_is_accessible(url):
    """
    If you see this test failing, find the offending dataset in the parametrization and move it to
    ``test_url_is_not_accessible`` and link an issue detailing the problem.
    """
    retry(lambda: assert_url_is_accessible(url))


@url_parametrization(
    stanford_cars,  # https://github.com/pytorch/vision/issues/7545
)
@pytest.mark.xfail
def test_url_is_not_accessible(url):
    """
    As the name implies, this test is the 'inverse' of ``test_url_is_accessible``. Since the download servers are
    beyond our control, some files might not be accessible for longer stretches of time. Still, we want to know if they
    come back up, or if we need to remove the download functionality of the dataset for good.

    If you see this test failing, find the offending dataset in the parametrization and move it to
    ``test_url_is_accessible``.
    """
    assert_url_is_accessible(url)
