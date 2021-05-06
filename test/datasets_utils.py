import contextlib
import functools
import importlib
import inspect
import itertools
import os
import pathlib
import random
import string
import unittest
import unittest.mock
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import PIL
import PIL.Image
import torch
import torchvision.datasets
import torchvision.io

from common_utils import get_tmp_dir, disable_console_output


__all__ = [
    "UsageError",
    "lazy_importer",
    "test_all_configs",
    "DatasetTestCase",
    "ImageDatasetTestCase",
    "VideoDatasetTestCase",
    "create_image_or_video_tensor",
    "create_image_file",
    "create_image_folder",
    "create_video_file",
    "create_video_folder",
    "create_random_string",
]


class UsageError(Exception):
    """Should be raised in case an error happens in the setup rather than the test."""


class LazyImporter:
    r"""Lazy importer for additional dependencies.

    Some datasets require additional packages that are no direct dependencies of torchvision. Instances of this class
    provide modules listed in MODULES as attributes. They are only imported when accessed.

    """
    MODULES = (
        "av",
        "lmdb",
        "pycocotools",
        "requests",
        "scipy.io",
        "scipy.sparse",
    )

    def __init__(self):
        modules = defaultdict(list)
        for module in self.MODULES:
            module, *submodules = module.split(".", 1)
            if submodules:
                modules[module].append(submodules[0])
            else:
                # This introduces the module so that it is known when we later iterate over the dictionary.
                modules.__missing__(module)

        for module, submodules in modules.items():
            # We need the quirky 'module=module' and submodules=submodules arguments to the lambda since otherwise the
            # lookup for these would happen at runtime rather than at definition. Thus, without it, every property
            # would try to import the last item in 'modules'
            setattr(
                type(self),
                module,
                property(lambda self, module=module, submodules=submodules: LazyImporter._import(module, submodules)),
            )

    @staticmethod
    def _import(package, subpackages):
        try:
            module = importlib.import_module(package)
        except ImportError as error:
            raise UsageError(
                f"Failed to import module '{package}'. "
                f"This probably means that the current test case needs '{package}' installed, "
                f"but it is not a dependency of torchvision. "
                f"You need to install it manually, for example 'pip install {package}'."
            ) from error

        for name in subpackages:
            importlib.import_module(f".{name}", package=package)

        return module


lazy_importer = LazyImporter()


def requires_lazy_imports(*modules):
    def outer_wrapper(fn):
        @functools.wraps(fn)
        def inner_wrapper(*args, **kwargs):
            for module in modules:
                getattr(lazy_importer, module.replace(".", "_"))
            return fn(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def test_all_configs(test):
    """Decorator to run test against all configurations.

    Add this as decorator to an arbitrary test to run it against all configurations. This includes
    :attr:`DatasetTestCase.DEFAULT_CONFIG` and :attr:`DatasetTestCase.ADDITIONAL_CONFIGS`.

    The current configuration is provided as the first parameter for the test:

    .. code-block::

        @test_all_configs()
        def test_foo(self, config):
            pass

    .. note::

        This will try to remove duplicate configurations. During this process it will not not preserve a potential
        ordering of the configurations or an inner ordering of a configuration.
    """

    def maybe_remove_duplicates(configs):
        try:
            return [dict(config_) for config_ in set(tuple(sorted(config.items())) for config in configs)]
        except TypeError:
            # A TypeError will be raised if a value of any config is not hashable, e.g. a list. In that case duplicate
            # removal would be a lot more elaborate and we simply bail out.
            return configs

    @functools.wraps(test)
    def wrapper(self):
        configs = []
        if self.DEFAULT_CONFIG is not None:
            configs.append(self.DEFAULT_CONFIG)
        if self.ADDITIONAL_CONFIGS is not None:
            configs.extend(self.ADDITIONAL_CONFIGS)

        if not configs:
            configs = [self._KWARG_DEFAULTS.copy()]
        else:
            configs = maybe_remove_duplicates(configs)

        for config in configs:
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
            get a reasonable default, that should work for most cases. Each entry of the sequence may be a tuple,
            to indicate multiple possible values.

    Optionally, you can overwrite the following class attributes:

        - DEFAULT_CONFIG (Dict[str, Any]): Config that will be used by default. If omitted, this defaults to all
            keyword arguments of the dataset minus ``transform``, ``target_transform``, ``transforms``, and
            ``download``. Overwrite this if you want to use a default value for a parameter for which the dataset does
            not provide one.
        - ADDITIONAL_CONFIGS (Sequence[Dict[str, Any]]): Additional configs that should be tested. Each dictionary can
            contain an arbitrary combination of dataset parameters that are **not** ``transform``, ``target_transform``,
            ``transforms``, or ``download``.
        - REQUIRED_PACKAGES (Iterable[str]): Additional dependencies to use the dataset. If these packages are not
            available, the tests are skipped.

    Additionally, you need to overwrite the ``inject_fake_data()`` method that provides the data that the tests rely on.
    The fake data should resemble the original data as close as necessary, while containing only few examples. During
    the creation of the dataset check-, download-, and extract-functions from ``torchvision.datasets.utils`` are
    disabled.

    Without further configuration, the testcase will test if

    1. the dataset raises a :class:`FileNotFoundError` or a :class:`RuntimeError` if the data files are not found or
       corrupted,
    2. the dataset inherits from `torchvision.datasets.VisionDataset`,
    3. the dataset can be turned into a string,
    4. the feature types of a returned example matches ``FEATURE_TYPES``,
    5. the number of examples matches the injected fake data, and
    6. the dataset calls ``transform``, ``target_transform``, or ``transforms`` if available when accessing data.

    Case 3. to 6. are tested against all configurations in ``CONFIGS``.

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

    DEFAULT_CONFIG = None
    ADDITIONAL_CONFIGS = None
    REQUIRED_PACKAGES = None

    # These keyword arguments are checked by test_transforms in case they are available in DATASET_CLASS.
    _TRANSFORM_KWARGS = {
        "transform",
        "target_transform",
        "transforms",
    }
    # These keyword arguments get a 'special' treatment and should not be set in DEFAULT_CONFIG or ADDITIONAL_CONFIGS.
    _SPECIAL_KWARGS = {
        *_TRANSFORM_KWARGS,
        "download",
    }

    # These fields are populated during setupClass() within _populate_private_class_attributes()

    # This will be a dictionary containing all keyword arguments with their respective default values extracted from
    # the dataset constructor.
    _KWARG_DEFAULTS = None
    # This will be a set of all _SPECIAL_KWARGS that the dataset constructor takes.
    _HAS_SPECIAL_KWARG = None

    # These functions are disabled during dataset creation in create_dataset().
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

    def dataset_args(self, tmpdir: str, config: Dict[str, Any]) -> Sequence[Any]:
        """Define positional arguments passed to the dataset.

        .. note::

            The default behavior is only valid if the dataset to be tested has ``root`` as the only required parameter.
            Otherwise you need to overwrite this method.

        Args:
            tmpdir (str): Path to a temporary directory. For most cases this acts as root directory for the dataset
                to be created and in turn also for the fake data injected here.
            config (Dict[str, Any]): Configuration that will be passed to the dataset constructor. It provides at least
                fields for all dataset parameters with default values.

        Returns:
            (Tuple[str]): ``tmpdir`` which corresponds to ``root`` for most datasets.
        """
        return (tmpdir,)

    def inject_fake_data(self, tmpdir: str, config: Dict[str, Any]) -> Union[int, Dict[str, Any]]:
        """Inject fake data for dataset into a temporary directory.

        During the creation of the dataset the download and extract logic is disabled. Thus, the fake data injected
        here needs to resemble the raw data, i.e. the state of the dataset directly after the files are downloaded and
        potentially extracted.

        Args:
            tmpdir (str): Path to a temporary directory. For most cases this acts as root directory for the dataset
                to be created and in turn also for the fake data injected here.
            config (Dict[str, Any]): Configuration that will be passed to the dataset constructor. It provides at least
                fields for all dataset parameters with default values.

        Needs to return one of the following:

            1. (int): Number of examples in the dataset to be created, or
            2. (Dict[str, Any]): Additional information about the injected fake data. Must contain the field
                ``"num_examples"`` that corresponds to the number of examples in the dataset to be created.
        """
        raise NotImplementedError("You need to provide fake data in order for the tests to run.")

    @contextlib.contextmanager
    def create_dataset(
        self,
        config: Optional[Dict[str, Any]] = None,
        inject_fake_data: bool = True,
        patch_checks: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[torchvision.datasets.VisionDataset, Dict[str, Any]]]:
        r"""Create the dataset in a temporary directory.

        The configuration passed to the dataset is populated to contain at least all parameters with default values.
        For this the following order of precedence is used:

        1. Parameters in :attr:`kwargs`.
        2. Configuration in :attr:`config`.
        3. Configuration in :attr:`~DatasetTestCase.DEFAULT_CONFIG`.
        4. Default parameters of the dataset.

        Args:
            config (Optional[Dict[str, Any]]): Configuration that will be used to create the dataset.
            inject_fake_data (bool): If ``True`` (default) inject the fake data with :meth:`.inject_fake_data` before
                creating the dataset.
            patch_checks (Optional[bool]): If ``True`` disable integrity check logic while creating the dataset. If
                omitted defaults to the same value as ``inject_fake_data``.
            **kwargs (Any): Additional parameters passed to the dataset. These parameters take precedence in case they
                overlap with ``config``.

        Yields:
            dataset (torchvision.dataset.VisionDataset): Dataset.
            info (Dict[str, Any]): Additional information about the injected fake data. See :meth:`.inject_fake_data`
                for details.
        """
        if patch_checks is None:
            patch_checks = inject_fake_data

        special_kwargs, other_kwargs = self._split_kwargs(kwargs)

        complete_config = self._KWARG_DEFAULTS.copy()
        if self.DEFAULT_CONFIG:
            complete_config.update(self.DEFAULT_CONFIG)
        if config:
            complete_config.update(config)
        if other_kwargs:
            complete_config.update(other_kwargs)

        if "download" in self._HAS_SPECIAL_KWARG and special_kwargs.get("download", False):
            # override download param to False param if its default is truthy
            special_kwargs["download"] = False

        patchers = self._patch_download_extract()
        if patch_checks:
            patchers.update(self._patch_checks())

        with get_tmp_dir() as tmpdir:
            args = self.dataset_args(tmpdir, complete_config)
            info = self._inject_fake_data(tmpdir, complete_config) if inject_fake_data else None

            with self._maybe_apply_patches(patchers), disable_console_output():
                dataset = self.DATASET_CLASS(*args, **complete_config, **special_kwargs)

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
        defaults = []
        for cls_ in cls.DATASET_CLASS.__mro__:
            if cls_ is torchvision.datasets.VisionDataset:
                break

            argspec = inspect.getfullargspec(cls_.__init__)

            if not argspec.defaults:
                continue

            defaults.append(
                {kwarg: default for kwarg, default in zip(argspec.args[-len(argspec.defaults):], argspec.defaults)}
            )

            if not argspec.varkw:
                break

        kwarg_defaults = dict()
        for config in reversed(defaults):
            kwarg_defaults.update(config)

        has_special_kwargs = set()
        for name in cls._SPECIAL_KWARGS:
            if name not in kwarg_defaults:
                continue

            del kwarg_defaults[name]
            has_special_kwargs.add(name)

        cls._KWARG_DEFAULTS = kwarg_defaults
        cls._HAS_SPECIAL_KWARG = has_special_kwargs

    @classmethod
    def _process_optional_public_class_attributes(cls):
        def check_config(config, name):
            special_kwargs = tuple(f"'{name}'" for name in cls._SPECIAL_KWARGS if name in config)
            if special_kwargs:
                raise UsageError(
                    f"{name} contains a value for the parameter(s) {', '.join(special_kwargs)}. "
                    f"These are handled separately by the test case and should not be set here. "
                    f"If you need to test some custom behavior regarding these parameters, "
                    f"you need to write a custom test (*not* test case), e.g. test_custom_transform()."
                )

        if cls.DEFAULT_CONFIG is not None:
            check_config(cls.DEFAULT_CONFIG, "DEFAULT_CONFIG")

        if cls.ADDITIONAL_CONFIGS is not None:
            for idx, config in enumerate(cls.ADDITIONAL_CONFIGS):
                check_config(config, f"CONFIGS[{idx}]")

        if cls.REQUIRED_PACKAGES:
            missing_pkgs = []
            for pkg in cls.REQUIRED_PACKAGES:
                try:
                    importlib.import_module(pkg)
                except ImportError:
                    missing_pkgs.append(f"'{pkg}'")

            if missing_pkgs:
                raise unittest.SkipTest(
                    f"The package(s) {', '.join(missing_pkgs)} are required to load the dataset "
                    f"'{cls.DATASET_CLASS.__name__}', but are not installed."
                )

    def _split_kwargs(self, kwargs):
        special_kwargs = kwargs.copy()
        other_kwargs = {key: special_kwargs.pop(key) for key in set(special_kwargs.keys()) - self._SPECIAL_KWARGS}
        return special_kwargs, other_kwargs

    def _inject_fake_data(self, tmpdir, config):
        info = self.inject_fake_data(tmpdir, config)
        if info is None:
            raise UsageError(
                "The method 'inject_fake_data' needs to return at least an integer indicating the number of "
                "examples for the current configuration."
            )
        elif isinstance(info, int):
            info = dict(num_examples=info)
        elif not isinstance(info, dict):
            raise UsageError(
                f"The additional information returned by the method 'inject_fake_data' must be either an "
                f"integer indicating the number of examples for the current configuration or a dictionary with "
                f"the same content. Got {type(info)} instead."
            )
        elif "num_examples" not in info:
            raise UsageError(
                "The information dictionary returned by the method 'inject_fake_data' must contain a "
                "'num_examples' field that holds the number of examples for the current configuration."
            )
        return info

    def _patch_download_extract(self):
        module = inspect.getmodule(self.DATASET_CLASS).__name__
        return {unittest.mock.patch(f"{module}.{function}") for function in self._DOWNLOAD_EXTRACT_FUNCTIONS}

    def _patch_checks(self):
        module = inspect.getmodule(self.DATASET_CLASS).__name__
        return {unittest.mock.patch(f"{module}.{function}", return_value=True) for function in self._CHECK_FUNCTIONS}

    @contextlib.contextmanager
    def _maybe_apply_patches(self, patchers):
        with contextlib.ExitStack() as stack:
            mocks = {}
            for patcher in patchers:
                with contextlib.suppress(AttributeError):
                    mocks[patcher.target] = stack.enter_context(patcher)
            yield mocks

    def test_not_found_or_corrupted(self):
        with self.assertRaises((FileNotFoundError, RuntimeError)):
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

            if len(self.FEATURE_TYPES) > 1:
                actual = len(example)
                expected = len(self.FEATURE_TYPES)
                self.assertEqual(
                    actual,
                    expected,
                    f"The number of the returned features does not match the the number of elements in FEATURE_TYPES: "
                    f"{actual} != {expected}",
                )
            else:
                example = (example,)

            for idx, (feature, expected_feature_type) in enumerate(zip(example, self.FEATURE_TYPES)):
                with self.subTest(idx=idx):
                    self.assertIsInstance(feature, expected_feature_type)

    @test_all_configs
    def test_num_examples(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertEqual(len(dataset), info["num_examples"])

    @test_all_configs
    def test_transforms(self, config):
        mock = unittest.mock.Mock(wraps=lambda *args: args[0] if len(args) == 1 else args)
        for kwarg in self._TRANSFORM_KWARGS:
            if kwarg not in self._HAS_SPECIAL_KWARG:
                continue

            mock.reset_mock()

            with self.subTest(kwarg=kwarg):
                with self.create_dataset(config, **{kwarg: mock}) as (dataset, _):
                    dataset[0]

                mock.assert_called()


class ImageDatasetTestCase(DatasetTestCase):
    """Abstract base class for image dataset testcases.

    - Overwrites the FEATURE_TYPES class attribute to expect a :class:`PIL.Image.Image` and an integer label.
    """

    FEATURE_TYPES = (PIL.Image.Image, int)

    @contextlib.contextmanager
    def create_dataset(
        self,
        config: Optional[Dict[str, Any]] = None,
        inject_fake_data: bool = True,
        patch_checks: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[torchvision.datasets.VisionDataset, Dict[str, Any]]]:
        with super().create_dataset(
            config=config,
            inject_fake_data=inject_fake_data,
            patch_checks=patch_checks,
            **kwargs,
        ) as (dataset, info):
            # PIL.Image.open() only loads the image meta data upfront and keeps the file open until the first access
            # to the pixel data occurs. Trying to delete such a file results in an PermissionError on Windows. Thus, we
            # force-load opened images.
            # This problem only occurs during testing since some tests, e.g. DatasetTestCase.test_feature_types open an
            # image, but never use the underlying data. During normal operation it is reasonable to assume that the
            # user wants to work with the image he just opened rather than deleting the underlying file.
            with self._force_load_images():
                yield dataset, info

    @contextlib.contextmanager
    def _force_load_images(self):
        open = PIL.Image.open

        def new(fp, *args, **kwargs):
            image = open(fp, *args, **kwargs)
            if isinstance(fp, (str, pathlib.Path)):
                image.load()
            return image

        with unittest.mock.patch("PIL.Image.open", new=new):
            yield


class VideoDatasetTestCase(DatasetTestCase):
    """Abstract base class for video dataset testcases.

    - Overwrites the 'FEATURE_TYPES' class attribute to expect two :class:`torch.Tensor` s for the video and audio as
      well as an integer label.
    - Overwrites the 'REQUIRED_PACKAGES' class attribute to require PyAV (``av``).
    - Adds the 'DEFAULT_FRAMES_PER_CLIP' class attribute. If no 'frames_per_clip' is provided by 'inject_fake_data()'
        and it is the last parameter without a default value in the dataset constructor, the value of the
        'DEFAULT_FRAMES_PER_CLIP' class attribute is appended to the output.
    """

    FEATURE_TYPES = (torch.Tensor, torch.Tensor, int)
    REQUIRED_PACKAGES = ("av",)

    DEFAULT_FRAMES_PER_CLIP = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_args = self._set_default_frames_per_clip(self.dataset_args)

    def _set_default_frames_per_clip(self, inject_fake_data):
        argspec = inspect.getfullargspec(self.DATASET_CLASS.__init__)
        args_without_default = argspec.args[1:-len(argspec.defaults)]
        frames_per_clip_last = args_without_default[-1] == "frames_per_clip"

        @functools.wraps(inject_fake_data)
        def wrapper(tmpdir, config):
            args = inject_fake_data(tmpdir, config)
            if frames_per_clip_last and len(args) == len(args_without_default) - 1:
                args = (*args, self.DEFAULT_FRAMES_PER_CLIP)

            return args

        return wrapper


def create_image_or_video_tensor(size: Sequence[int]) -> torch.Tensor:
    r"""Create a random uint8 tensor.

    Args:
        size (Sequence[int]): Size of the tensor.
    """
    return torch.randint(0, 256, size, dtype=torch.uint8)


def create_image_file(
    root: Union[pathlib.Path, str], name: Union[pathlib.Path, str], size: Union[Sequence[int], int] = 10, **kwargs: Any
) -> pathlib.Path:
    """Create an image file from random data.

    Args:
        root (Union[str, pathlib.Path]): Root directory the image file will be placed in.
        name (Union[str, pathlib.Path]): Name of the image file.
        size (Union[Sequence[int], int]): Size of the image that represents the ``(num_channels, height, width)``. If
            scalar, the value is used for the height and width. If not provided, three channels are assumed.
        kwargs (Any): Additional parameters passed to :meth:`PIL.Image.Image.save`.

    Returns:
        pathlib.Path: Path to the created image file.
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
    file = pathlib.Path(root) / name

    # torch (num_channels x height x width) -> PIL (width x height x num_channels)
    image = image.permute(2, 1, 0)
    # For grayscale images PIL doesn't use a channel dimension
    if image.shape[2] == 1:
        image = torch.squeeze(image, 2)
    PIL.Image.fromarray(image.numpy()).save(file, **kwargs)
    return file


def create_image_folder(
    root: Union[pathlib.Path, str],
    name: Union[pathlib.Path, str],
    file_name_fn: Callable[[int], str],
    num_examples: int,
    size: Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]] = None,
    **kwargs: Any,
) -> List[pathlib.Path]:
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

    Returns:
        List[pathlib.Path]: Paths to all created image files.

    .. seealso::

        - :func:`create_image_file`
    """
    if size is None:

        def size(idx: int) -> Tuple[int, int, int]:
            num_channels = 3
            height, width = torch.randint(3, 11, size=(2,), dtype=torch.int).tolist()
            return (num_channels, height, width)

    root = pathlib.Path(root) / name
    os.makedirs(root, exist_ok=True)

    return [
        create_image_file(root, file_name_fn(idx), size=size(idx) if callable(size) else size, **kwargs)
        for idx in range(num_examples)
    ]


@requires_lazy_imports("av")
def create_video_file(
    root: Union[pathlib.Path, str],
    name: Union[pathlib.Path, str],
    size: Union[Sequence[int], int] = (1, 3, 10, 10),
    fps: float = 25,
    **kwargs: Any,
) -> pathlib.Path:
    """Create an video file from random data.

    Args:
        root (Union[str, pathlib.Path]): Root directory the video file will be placed in.
        name (Union[str, pathlib.Path]): Name of the video file.
        size (Union[Sequence[int], int]): Size of the video that represents the
            ``(num_frames, num_channels, height, width)``. If scalar, the value is used for the height and width.
            If not provided, ``num_frames=1`` and ``num_channels=3`` are assumed.
        fps (float): Frame rate in frames per second.
        kwargs (Any): Additional parameters passed to :func:`torchvision.io.write_video`.

    Returns:
        pathlib.Path: Path to the created image file.

    Raises:
        UsageError: If PyAV is not available.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(size) == 2:
        size = (3, *size)
    if len(size) == 3:
        size = (1, *size)
    if len(size) != 4:
        raise UsageError(
            f"The 'size' argument should either be an int or a sequence of length 2, 3, or 4. Got {len(size)} instead"
        )

    video = create_image_or_video_tensor(size)
    file = pathlib.Path(root) / name
    torchvision.io.write_video(str(file), video.permute(0, 2, 3, 1), fps, **kwargs)
    return file


@requires_lazy_imports("av")
def create_video_folder(
    root: Union[str, pathlib.Path],
    name: Union[str, pathlib.Path],
    file_name_fn: Callable[[int], str],
    num_examples: int,
    size: Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]] = None,
    fps=25,
    **kwargs,
) -> List[pathlib.Path]:
    """Create a folder of random videos.

    Args:
        root (Union[str, pathlib.Path]): Root directory the video folder will be placed in.
        name (Union[str, pathlib.Path]): Name of the video folder.
        file_name_fn (Callable[[int], str]): Should return a file name if called with the file index.
        num_examples (int): Number of videos to create.
        size (Optional[Union[Sequence[int], int, Callable[[int], Union[Sequence[int], int]]]]): Size of the videos. If
            callable, will be called with the index of the corresponding file. If omitted, a random even height and
            width between 4 and 10 pixels is selected on a per-video basis.
        fps (float): Frame rate in frames per second.
        kwargs (Any): Additional parameters passed to :func:`create_video_file`.

    Returns:
        List[pathlib.Path]: Paths to all created video files.

    Raises:
        UsageError: If PyAV is not available.

    .. seealso::

        - :func:`create_video_file`
    """
    if size is None:

        def size(idx):
            num_frames = 1
            num_channels = 3
            # The 'libx264' video codec, which is the default of torchvision.io.write_video, requires the height and
            # width of the video to be divisible by 2.
            height, width = (torch.randint(2, 6, size=(2,), dtype=torch.int) * 2).tolist()
            return (num_frames, num_channels, height, width)

    root = pathlib.Path(root) / name
    os.makedirs(root, exist_ok=True)

    return [
        create_video_file(root, file_name_fn(idx), size=size(idx) if callable(size) else size, **kwargs)
        for idx in range(num_examples)
    ]


def create_random_string(length: int, *digits: str) -> str:
    """Create a random string.

    Args:
        length (int): Number of characters in the generated string.
        *characters (str): Characters to sample from. If omitted defaults to :attr:`string.ascii_lowercase`.
    """
    if not digits:
        digits = string.ascii_lowercase
    else:
        digits = "".join(itertools.chain(*digits))

    return "".join(random.choice(digits) for _ in range(length))
