import contextlib
import functools
import importlib
import inspect
import itertools
import unittest
import unittest.mock
from typing import Any, Iterator, Sequence, Tuple, Union

from PIL import Image

import torchvision.datasets

from datasets_utils import tmpdir, disable_console_output


__all__ = ["DatasetTestCase", "ImageDatasetTestCase", "VideoDatasetTestCase", "test_all_configs"]


class UsageError(RuntimeError):
    """Should be raised instead of a generic ``RuntimeError`` in case a test case is not correctly configured."""


# As of Python 3.7 this is provided by contextlib
# https://docs.python.org/3.7/library/contextlib.html#contextlib.nullcontext
# TODO: If the minimum Python requirement is >= 3.7, replace this
@contextlib.contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def test_all_configs(test):
    """Decorator to run test against all configurations.

    Add this as decorator to an arbitrary test to run it against all configurations. The current configuration is
    provided as the first parameter:

    .. code-block::

        @test_all_configs
        def test_foo(self, config):
            pass
    """

    @functools.wraps(test)
    def wrapper(self):
        for config in self.CONFIGS:
            with self.subTest(**config):
                test(self, config)

    return wrapper


class DatasetTestCase(unittest.TestCase):
    """Abstract base class for all dataset testcases.

    You have to overwrite the following class attributes:

        - DATASET_CLASS (torchvision.datasets.VisionDataset): Class of dataset to be tested.
        - FEATURE_TYPES (Sequence[Any]): Types of the elements returned by index access of the dataset. Instead of
            providing these manually, you can instead subclass ``ImageDatasetTestCase`` or ``VideoDatasetTestCase```to
            get a reasonable default, that should work for most cases.

    Optionally, you can overwrite the following class attributes:

        - CONFIGS (Sequence[Dict[str, Any]]): Additional configs that should be tested. Each dictonary can contain an
            arbitrary combination of dataset parameters that are **not** ``transform``, ``target_transform``,
            ``transforms``, or ``download``. The first element will be used as default configuration.
        - REQUIRED_PACKAGES (Iterable[str]): Additional dependencies to use the dataset. If these packages are not
            available, the tests are skipped.

    Additionally, you need to overwrite the ``inject_fake_data()`` method that provides the data that the tests rely on.
    The fake data should resemble the original data as close as necessary, while containing only few examples. During
    the creation of the dataset check-, download-, and extract-functions from ``torchvision.datasets.utils`` are
    disabled.

    Without further configuration, the testcase will test if

    1. the dataset raises a ``RuntimeError`` if the data files are not found,
    2. the dataset inherits from `torchvision.datasets.VisionDataset`,
    3. the dataset can be turned into a string,
    4. the feature types of a returned example matches ``FEATURE_TYPES``, and
    5. the number of examples matches the injected fake data.

    Case 3., 4., and 5. are tested against all configurations in ``CONFIGS``.

    To add dataset-specific tests, create a new method that takes no arguments with ``test_`` as a name prefix:

    .. code-block::

        def test_foo(self):
            pass

    If you want to run the test against all configs, add the ``@test_all_configs`` decorator to the definition and
    accept a single argument:

    .. code-block::

        @test_all_configs
        def test_bar(self, config):
            pass

    Within the test you can use the ``create_dataset()`` method that yields the dataset as well as additional information
    provided by the ``ìnject_fake_data()`` method:

    .. code-block::

        def test_baz(self):
            with self.create_dataset() as (dataset, info):
                pass
    """

    DATASET_CLASS = None
    FEATURE_TYPES = None

    CONFIGS = None
    REQUIRED_PACKAGES = None

    _SPECIAL_KWARGS = {
        "transform",
        "target_transform",
        "transforms",
        "download",
    }
    _HAS_SPECIAL_KWARG = None

    _CHECK_FUNCTIONS = {
        "check_md5",
        "check_integrity",
    }
    _DOWNLOAD_EXTRACT_FUNCTIONS = {
        "download_url",
        "download_file_from_google_drive",
        "extract_archive",
        "download_and_extract_archive",
    }

    def inject_fake_data(self, root: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inject fake data into the root of the dataset.

        Args:
            root (str): Root of the dataset.
            config (Dict[str, Any]): Configuration that will be used to create the dataset.

        Returns:
            info (Dict[str, Any]): Additional information about the injected fake data. Must contain the field
                ``"num_examples"`` that corresponds to the length of the dataset to be created.
        """
        raise NotImplementedError("You need to provide fake data in order for the tests to run.")

    @contextlib.contextmanager
    def create_dataset(
        self,
        config: Optional[Dict[str, Any]] = None,
        inject_fake_data: bool = True,
        disable_download_extract: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[torchvision.datasets.VisionDataset, Dict[str, Any]]]:
        r"""Create the dataset in a temporary directory.

        Args:
            config (Optional[Dict[str, Any]]): Configuration that will be used to create the dataset. If omitted, the
                default configuration is used.
            inject_fake_data (bool): If ``True`` (default) inject the fake data with :meth:`.inject_fake_data` before
                creating the dataset.
            disable_download_extract (Optional[bool]): If ``True`` disable download and extract logic while creating
                the dataset. If ``None`` (default) this takes the same value as ``inject_fake_data``.
            **kwargs (Any): Additional parameters passed to the dataset. These parameters take precedence in case they
                overlap with ``config``.

        Yields:
            dataset (torchvision.dataset.VisionDataset): Dataset.
            info (Dict[str, Any]): Additional information about the injected fake data. See :meth:`.inject_fake_data`
                for details.
        """
        if config is None:
            config = self.CONFIGS[0]

        special_kwargs, other_kwargs = self._split_kwargs(kwargs)
        config.update(other_kwargs)

        if disable_download_extract is None:
            disable_download_extract = inject_fake_data

        with tmpdir() as root:
            info = self.inject_fake_data(root, config) if inject_fake_data else None
            if info is None or "num_examples" not in info:
                raise UsageError(
                    "The method 'inject_fake_data' needs to return a dictionary that contains at least a "
                    "'num_examples' field."
                )

            cm = self._disable_download_extract if disable_download_extract else nullcontext
            with cm(special_kwargs), disable_console_output():
                dataset = self.DATASET_CLASS(root, **config, **special_kwargs)

            yield dataset, info

    @classmethod
    def setUpClass(cls):
        cls._verify_required_public_class_attributes()
        cls._populate_private_class_attributes()
        cls._process_optional_public_class_attributes()
        super().setUpClass()

    @classmethod
    def _verify_required_public_class_attributes(cls):
        if cls.DATASET_CLASS is None:
            raise UsageError(
                "The class attribute 'DATASET_CLASS' needs to be overwritten. "
                "It should contain the class of the dataset to be tested."
            )
        if cls.FEATURE_TYPES is None:
            raise UsageError(
                "The class attribute 'FEATURE_TYPES' needs to be overwritten. "
                "It should contain a sequence of types that the dataset returns when accessed by index."
            )

    @property
    @classmethod
    def _argspec(cls):
        return inspect.getfullargspec(cls.DATASET_CLASS.__init__)

    @property
    @classmethod
    def _name(cls):
        return cls.DATASET_CLASS.__name__

    @classmethod
    def _populate_private_class_attributes(cls):
        cls._HAS_SPECIAL_KWARG = {name: name in cls._argspec.args for name in cls._SPECIAL_KWARGS}

    @classmethod
    def _process_optional_public_class_attributes(cls):
        argspec = cls._argspec
        if cls.CONFIGS is None:
            config = {
                kwarg: default
                for kwarg, default in zip(argspec.args[-len(argspec.defaults) :], argspec.defaults)
                if kwarg not in cls._SPECIAL_KWARGS
            }
            cls.CONFIGS = (config,)

        if cls.REQUIRED_PACKAGES is not None:
            try:
                for pkg in cls.REQUIRED_PACKAGES:
                    importlib.import_module(pkg)
            except ImportError as error:
                raise unittest.SkipTest(
                    f"The package '{error.name}' is required to load the dataset '{cls._name}' but is not installed."
                )

    def _split_kwargs(self, kwargs):
        special_kwargs = kwargs.copy()
        other_kwargs = {key: special_kwargs.pop(key) for key in set(special_kwargs.keys()) - self._SPECIAL_KWARGS}
        return special_kwargs, other_kwargs

    @contextlib.contextmanager
    def _disable_download_extract(self, special_kwargs):
        inject_download_kwarg = self._HAS_SPECIAL_KWARG["download"] and "download" not in special_kwargs
        if inject_download_kwarg:
            special_kwargs["download"] = False

        module = inspect.getmodule(self.DATASET_CLASS).__name__
        with contextlib.ExitStack() as stack:
            mocks = {}
            for function, kwargs in itertools.chain(
                zip(self._CHECK_FUNCTIONS, [dict(return_value=True)] * len(self._CHECK_FUNCTIONS)),
                zip(self._DOWNLOAD_EXTRACT_FUNCTIONS, [dict()] * len(self._DOWNLOAD_EXTRACT_FUNCTIONS)),
            ):
                with contextlib.suppress(AttributeError):
                    patcher = unittest.mock.patch(f"{module}.{function}", **kwargs)
                    mocks[function] = stack.enter_context(patcher)

            try:
                yield mocks
            finally:
                if inject_download_kwarg:
                    del special_kwargs["download"]

    def test_not_found(self):
        with self.assertRaises(RuntimeError):
            with self.create_dataset(inject_fake_data=False):
                pass

    def test_smoke(self, config):
        with self.create_dataset(config) as (dataset, _):
            self.assertIsInstance(dataset, torchvision.datasets.VisionDataset)

    @test_all_configs
    def test_str_smoke(self, config):
        with self.create_dataset(config) as (dataset, _):
            self.assertIsInstance(str(dataset), str)

    @test_all_configs
    def test_feature_types(self, config):
        with self.create_dataset(config) as (dataset, _):
            example = dataset[0]

            actual = len(example)
            expected = len(self.FEATURE_TYPES)
            self.assertEqual(
                actual,
                expected,
                f"The number of the returned features does not match the the number of elements in in FEATURE_TYPES: "
                f"{actual} != {expected}",
            )

            for idx, (feature, expected_feature_type) in enumerate(zip(example, self.FEATURE_TYPES)):
                with self.subTest(idx=idx):
                    self.assertIsInstance(feature, expected_feature_type)

    @test_all_configs
    def test_num_examples(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertEqual(len(dataset), info["num_examples"])


class ImageDatasetTestCase(DatasetTestCase):
    FEATURE_TYPES = (Image.Image, int)


class VideoDatasetTestCase(DatasetTestCase):
    FEATURE_TYPES = (torch.Tensor, torch.Tensor, int)
    REQUIRED_PACKAGES = ("av",)
