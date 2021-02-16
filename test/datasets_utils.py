import contextlib
import functools
import importlib
import inspect
import itertools
import os
import pathlib
import unittest
import unittest.mock
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import PIL.Image

import torch
import torchvision.datasets

from common_utils import get_tmp_dir, disable_console_output

try:
    from torchvision.io import write_video

    PYAV_AVAILABLE = True

except ImportError:
    write_video = None
    PYAV_AVAILABLE = False


__all__ = [
    "UsageError",
    "test_all_configs",
    "DatasetTestCase",
    "ImageDatasetTestCase",
    "VideoDatasetTestCase",
    "create_image_or_video_tensor",
    "create_image_file",
    "create_image_folder",
    "create_video_file",
    "create_video_folder",
]


class UsageError(RuntimeError):
    """Should be raised in case an error happens in the setup rather than the test."""


def requires_pyav(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not PYAV_AVAILABLE:
            raise UsageError("PyAV (av) is required but not available.")

        return fn(*args, **kwargs)

    return wrapper


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


def combinations_grid(**kwargs):
    """Creates a grid of input combinations.

    Each element in the returned sequence is a dictionary containing one possible combination as values.

    Example:
        >>> combinations_grid(foo=("bar", "baz"), spam=("eggs", "ham"))
        [
            {'foo': 'bar', 'spam': 'eggs'},
            {'foo': 'bar', 'spam': 'ham'},
            {'foo': 'baz', 'spam': 'eggs'},
            {'foo': 'baz', 'spam': 'ham'}
        ]
    """
    return [dict(zip(kwargs.keys(), values)) for values in itertools.product(*kwargs.values())]


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

    Within the test you can use the ``create_dataset()`` method that yields the dataset as well as additional
    information provided by the ``ìnject_fake_data()`` method:

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

        with get_tmp_dir() as root:
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

    @classmethod
    def _populate_private_class_attributes(cls):
        argspec = inspect.getfullargspec(cls.DATASET_CLASS.__init__)
        cls._HAS_SPECIAL_KWARG = {name: name in argspec.args for name in cls._SPECIAL_KWARGS}

    @classmethod
    def _process_optional_public_class_attributes(cls):
        argspec = inspect.getfullargspec(cls.DATASET_CLASS.__init__)
        if cls.CONFIGS is None:
            config = {
                kwarg: default
                for kwarg, default in zip(argspec.args[-len(argspec.defaults):], argspec.defaults)
                if kwarg not in cls._SPECIAL_KWARGS
            }
            cls.CONFIGS = (config,)

        if cls.REQUIRED_PACKAGES is not None:
            try:
                for pkg in cls.REQUIRED_PACKAGES:
                    importlib.import_module(pkg)
            except ImportError as error:
                raise unittest.SkipTest(
                    f"The package '{error.name}' is required to load the dataset '{cls.DATASET_CLASS.__name__}' but is "
                    f"not installed."
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

    def test_smoke(self):
        with self.create_dataset() as (dataset, _):
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
    """Abstract base class for image dataset testcases.

    - Overwrites the FEATURE_TYPES class attribute to expect a :class:`PIL.Image.Image` and an integer label.
    """

    FEATURE_TYPES = (PIL.Image.Image, int)


class VideoDatasetTestCase(DatasetTestCase):
    """Abstract base class for video dataset testcases.

    - Overwrites the FEATURE_TYPES class attribute to expect two :class:`torch.Tensor` s for the video and audio as
      well as an integer label.
    - Overwrites the REQUIRED_PACKAGES class attribute to require PyAV (``av``).
    """

    FEATURE_TYPES = (torch.Tensor, torch.Tensor, int)
    REQUIRED_PACKAGES = ("av",)


def create_image_or_video_tensor(size: Sequence[int]) -> torch.Tensor:
    r"""Create a random uint8 tensor.

    Args:
        size (Sequence[int]): Size of the tensor.
    """
    return torch.randint(0, 256, size, dtype=torch.uint8)


def create_image_file(
    root: Union[pathlib.Path, str], name: Union[pathlib.Path, str], size: Union[Sequence[int], int] = 10, **kwargs: Any
) -> None:
    """Create an image file from random data.

    Args:
        root (Union[str, pathlib.Path]): Root directory the image file will be placed in.
        name (Union[str, pathlib.Path]): Name of the image file.
        size (Union[Sequence[int], int]): Size of the image that represents the ``(num_channels, height, width)``. If
            scalar, the value is used for the height and width. If not provided, three channels are assumed.
        kwargs (Any): Additional parameters passed to :meth:`PIL.Image.Image.save`.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(size) == 2:
        size = (3, *size)
    if len(size) != 3:
        raise UsageError(
            f"The 'size' argument should either be an int or a sequence of length 2 or 3. Got {len(size)} instead"
        )

    image = create_image_or_video_tensor(size)
    PIL.Image.fromarray(image.permute(2, 1, 0).numpy()).save(pathlib.Path(root) / name)


def create_image_folder(
    root: Union[pathlib.Path, str],
    name: Union[pathlib.Path, str],
    file_name_fn: Callable[[int], str],
    num_examples: int,
    size: Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]] = None,
    **kwargs: Any,
):
    """Create a folder of random images.

    Args:
        root (Union[str, pathlib.Path]): Root directory the image folder will be placed in.
        name (Union[str, pathlib.Path]): Name of the image folder.
        file_name_fn (Callable[[int], str]): Should return a file name if called with the file index.
        num_examples (int): Number of images to create.
        size (Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]]): Size of the images. If
            callable, will be called with the index of the corresponding file. If omitted, a random height and width
            between 3 and 10 pixels is selected on a per-image basis.
        kwargs (Any): Additional parameters passed to :func:`create_image_file`.


    .. seealso::

        - :func:`create_image_file`
    """
    if size is None:

        def size(idx: int) -> Tuple[int, int, int]:
            num_channels = 3
            height, width = torch.randint(3, 11, size=(2,), dtype=torch.int).tolist()
            return (num_channels, height, width)

    root = pathlib.Path(root) / name
    os.makedirs(root)

    for idx in range(num_examples):
        create_image_file(root, file_name_fn(idx), size=size(idx) if callable(size) else size, **kwargs)


@requires_pyav
def create_video_file(
    root: Union[pathlib.Path, str],
    name: Union[pathlib.Path, str],
    size: Union[Sequence[int], int] = (25, 3, 10, 10),
    fps: float = 25,
    **kwargs: Any,
) -> None:
    """Create an video file from random data.

    Args:
        root (Union[str, pathlib.Path]): Root directory the video file will be placed in.
        name (Union[str, pathlib.Path]): Name of the video file.
        size (Union[Sequence[int], int]): Size of the video that represents the
            ``(length, num_channels, height, width)``. If scalar, the value is used for the height and width.
            If not provided, three channels are assumed. If not provided, the length is set to one second.
        fps (float): Frame rate in frames per second.
        kwargs (Any): Additional parameters passed to :func:`torchvision.io.write_video`.

    Raises:
        UsageError: If PyAV is not available.
    """
    if not PYAV_AVAILABLE:
        raise PyAVNotAvailableError

    if isinstance(size, int):
        size = (size, size)
    if len(size) == 2:
        size = (3, *size)
    if len(size) == 3:
        size = (fps, *size)
    if len(size) != 4:
        raise UsageError(
            f"The 'size' argument should either be an int or a sequence of length 2, 3, or 4. Got {len(size)} instead"
        )

    video = create_image_or_video_tensor(size)
    write_video(str(pathlib.Path(root) / name), video.permute(0, 2, 3, 1), fps, **kwargs)


@requires_pyav
def create_video_folder(
    root: Union[str, pathlib.Path],
    name: Union[str, pathlib.Path],
    file_name_fn: Callable[[int], str],
    num_examples: int,
    size: Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]] = None,
    fps=25,
    **kwargs,
):
    """Create a folder of random videos.

    Args:
        root (Union[str, pathlib.Path]): Root directory the image folder will be placed in.
        name (Union[str, pathlib.Path]): Name of the image folder.
        file_name_fn (Callable[[int], str]): Should return a file name if called with the file index.
        num_examples (int): Number of images to create.
        size (Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]]): Size of the images. If
            callable, will be called with the index of the corresponding file. If omitted, a random length between 0.5
            and 1.5 seconds as well as random even height and width between 4 and 10 pixels are selected on a
            per-video basis.
        fps (float): Frame rate in frames per second.
        kwargs (Any): Additional parameters passed to :func:`create_video_file`.

    Raises:
        UsageError: If PyAV is not available.

    .. seealso::

        - :func:`create_video_file`
    """
    if size is None:

        def size(idx):
            length = int((torch.rand(()).item() + 0.5) * fps)
            num_channels = 3
            # The 'libx264' video codec, which is the default of torchvision.io.write_video, requires the height and
            # width of the video to be divisible by 2.
            height, width = (torch.randint(2, 6, size=(2,), dtype=torch.int) * 2).tolist()
            return (length, num_channels, height, width)

    root = pathlib.Path(root) / name
    os.makedirs(root)

    for idx in range(num_examples):
        create_video_file(root, file_name_fn(idx), size=size(idx) if callable(size) else size)
