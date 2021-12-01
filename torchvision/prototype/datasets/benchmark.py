# type: ignore

import argparse
import collections.abc
import contextlib
import inspect
import itertools
import os
import os.path
import pathlib
import shutil
import sys
import tempfile
import time
import unittest.mock
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader_experimental import DataLoader2
from torchvision import datasets as legacy_datasets
from torchvision.datasets.utils import extract_archive
from torchvision.prototype import datasets as new_datasets
from torchvision.transforms import PILToTensor


def main(
    name,
    *,
    variant=None,
    legacy=True,
    new=True,
    start=True,
    iteration=True,
    num_starts=3,
    num_samples=10_000,
    temp_root=None,
    num_workers=0,
):
    benchmarks = [
        benchmark
        for benchmark in DATASET_BENCHMARKS
        if benchmark.name == name and (variant is None or benchmark.variant == variant)
    ]
    if not benchmarks:
        msg = f"No DatasetBenchmark available for dataset '{name}'"
        if variant is not None:
            msg += f" and variant '{variant}'"
        raise ValueError(msg)

    for benchmark in benchmarks:
        print("#" * 80)
        print(f"{benchmark.name}" + (f" ({benchmark.variant})" if benchmark.variant is not None else ""))

        if legacy and start:
            print(
                "legacy",
                "cold_start",
                Measurement.time(benchmark.legacy_cold_start(temp_root, num_workers=num_workers), number=num_starts),
            )
            print(
                "legacy",
                "warm_start",
                Measurement.time(benchmark.legacy_warm_start(temp_root, num_workers=num_workers), number=num_starts),
            )

        if legacy and iteration:
            print(
                "legacy",
                "iteration",
                Measurement.iterations_per_time(
                    benchmark.legacy_iteration(temp_root, num_workers=num_workers, num_samples=num_samples)
                ),
            )

        if new and start:
            print(
                "new",
                "cold_start",
                Measurement.time(benchmark.new_cold_start(num_workers=num_workers), number=num_starts),
            )

        if new and iteration:
            print(
                "new",
                "iteration",
                Measurement.iterations_per_time(
                    benchmark.new_iteration(num_workers=num_workers, num_samples=num_samples)
                ),
            )


class DatasetBenchmark:
    def __init__(
        self,
        name: str,
        *,
        variant=None,
        legacy_cls=None,
        new_config=None,
        legacy_config_map=None,
        legacy_special_options_map=None,
        prepare_legacy_root=None,
    ):
        self.name = name
        self.variant = variant

        self.new_raw_dataset = new_datasets._api.find(name)
        self.legacy_cls = legacy_cls or self._find_legacy_cls()

        if new_config is None:
            new_config = self.new_raw_dataset.default_config
        elif isinstance(new_config, dict):
            new_config = self.new_raw_dataset.info.make_config(**new_config)
        self.new_config = new_config

        self.legacy_config_map = legacy_config_map
        self.legacy_special_options_map = legacy_special_options_map or self._legacy_special_options_map
        self.prepare_legacy_root = prepare_legacy_root

    def new_dataset(self, *, num_workers=0):
        return DataLoader2(new_datasets.load(self.name, **self.new_config), num_workers=num_workers)

    def new_cold_start(self, *, num_workers):
        def fn(timer):
            with timer:
                dataset = self.new_dataset(num_workers=num_workers)
                next(iter(dataset))

        return fn

    def new_iteration(self, *, num_samples, num_workers):
        def fn(timer):
            dataset = self.new_dataset(num_workers=num_workers)
            num_sample = 0
            with timer:
                for _ in dataset:
                    num_sample += 1
                    if num_sample == num_samples:
                        break

            return num_sample

        return fn

    def suppress_output(self):
        @contextlib.contextmanager
        def context_manager():
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        return context_manager()

    def legacy_dataset(self, root, *, num_workers=0, download=None):
        legacy_config = self.legacy_config_map(self, root) if self.legacy_config_map else dict()

        special_options = self.legacy_special_options_map(self)
        if "download" in special_options and download is not None:
            special_options["download"] = download

        with self.suppress_output():
            return DataLoader(
                self.legacy_cls(legacy_config.pop("root", str(root)), **legacy_config, **special_options),
                shuffle=True,
                num_workers=num_workers,
            )

    @contextlib.contextmanager
    def patch_download_and_integrity_checks(self):
        patches = [
            ("download_url", dict()),
            ("download_file_from_google_drive", dict()),
            ("check_integrity", dict(new=lambda path, md5=None: os.path.isfile(path))),
        ]
        dataset_module = sys.modules[self.legacy_cls.__module__]
        utils_module = legacy_datasets.utils
        with contextlib.ExitStack() as stack:
            for name, patch_kwargs in patches:
                patch_module = dataset_module if name in dir(dataset_module) else utils_module
                stack.enter_context(unittest.mock.patch(f"{patch_module.__name__}.{name}", **patch_kwargs))

            yield stack

    def _find_resource_file_names(self):
        info = self.new_raw_dataset.info
        valid_options = info._valid_options

        file_names = set()
        for options in (
            dict(zip(valid_options.keys(), values)) for values in itertools.product(*valid_options.values())
        ):
            resources = self.new_raw_dataset.resources(info.make_config(**options))
            file_names.update([resource.file_name for resource in resources])

        return file_names

    @contextlib.contextmanager
    def legacy_root(self, temp_root):
        new_root = pathlib.Path(new_datasets.home()) / self.name
        legacy_root = pathlib.Path(tempfile.mkdtemp(dir=temp_root))

        if os.stat(new_root).st_dev != os.stat(legacy_root).st_dev:
            warnings.warn(
                "The temporary root directory for the legacy dataset was created on a different storage device than "
                "the raw data that is used by the new dataset. If the devices have different I/O stats, this will "
                "distort the benchmark. You can use the '--temp-root' flag to relocate the root directory of the "
                "temporary directories.",
                RuntimeWarning,
            )

        try:
            for file_name in self._find_resource_file_names():
                (legacy_root / file_name).symlink_to(new_root / file_name)

            if self.prepare_legacy_root:
                self.prepare_legacy_root(self, legacy_root)

            with self.patch_download_and_integrity_checks():
                yield legacy_root
        finally:
            shutil.rmtree(legacy_root)

    def legacy_cold_start(self, temp_root, *, num_workers):
        def fn(timer):
            with self.legacy_root(temp_root) as root:
                with timer:
                    dataset = self.legacy_dataset(root, num_workers=num_workers)
                    next(iter(dataset))

        return fn

    def legacy_warm_start(self, temp_root, *, num_workers):
        def fn(timer):
            with self.legacy_root(temp_root) as root:
                self.legacy_dataset(root, num_workers=num_workers)
                with timer:
                    dataset = self.legacy_dataset(root, num_workers=num_workers, download=False)
                    next(iter(dataset))

        return fn

    def legacy_iteration(self, temp_root, *, num_samples, num_workers):
        def fn(timer):
            with self.legacy_root(temp_root) as root:
                dataset = self.legacy_dataset(root, num_workers=num_workers)
                with timer:
                    for num_sample, _ in enumerate(dataset, 1):
                        if num_sample == num_samples:
                            break

            return num_sample

        return fn

    def _find_legacy_cls(self):
        legacy_clss = {
            name.lower(): dataset_class
            for name, dataset_class in legacy_datasets.__dict__.items()
            if isinstance(dataset_class, type) and issubclass(dataset_class, legacy_datasets.VisionDataset)
        }
        try:
            return legacy_clss[self.name]
        except KeyError as error:
            raise RuntimeError(
                f"Can't determine the legacy dataset class for '{self.name}' automatically. "
                f"Please set the 'legacy_cls' keyword argument manually."
            ) from error

    _SPECIAL_KWARGS = {
        "transform",
        "target_transform",
        "transforms",
        "download",
    }

    @staticmethod
    def _legacy_special_options_map(benchmark):
        available_parameters = set()

        for cls in benchmark.legacy_cls.__mro__:
            if cls is legacy_datasets.VisionDataset:
                break

            available_parameters.update(inspect.signature(cls.__init__).parameters)

        available_special_kwargs = benchmark._SPECIAL_KWARGS.intersection(available_parameters)

        special_options = dict()

        if "download" in available_special_kwargs:
            special_options["download"] = True

        if "transform" in available_special_kwargs:
            special_options["transform"] = PILToTensor()
            if "target_transform" in available_special_kwargs:
                special_options["target_transform"] = torch.tensor
        elif "transforms" in available_special_kwargs:
            special_options["transforms"] = JointTransform(PILToTensor(), PILToTensor())

        return special_options


class Measurement:
    @classmethod
    def time(cls, fn, *, number):
        results = Measurement._timeit(fn, number=number)
        times = torch.tensor(tuple(zip(*results))[1])
        return cls._format(times, unit="s")

    @classmethod
    def iterations_per_time(cls, fn):
        num_samples, time = Measurement._timeit(fn, number=1)[0]
        iterations_per_second = torch.tensor(num_samples) / torch.tensor(time)
        return cls._format(iterations_per_second, unit="it/s")

    class Timer:
        def __init__(self):
            self._start = None
            self._stop = None

        def __enter__(self):
            self._start = time.perf_counter()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._stop = time.perf_counter()

        @property
        def delta(self):
            if self._start is None:
                raise RuntimeError()
            elif self._stop is None:
                raise RuntimeError()
            return self._stop - self._start

    @classmethod
    def _timeit(cls, fn, number):
        results = []
        for _ in range(number):
            timer = cls.Timer()
            output = fn(timer)
            results.append((output, timer.delta))
        return results

    @classmethod
    def _format(cls, measurements, *, unit):
        measurements = torch.as_tensor(measurements).to(torch.float64).flatten()
        if measurements.numel() == 1:
            # TODO format that into engineering format
            return f"{float(measurements):.3f} {unit}"

        mean, std = Measurement._compute_mean_and_std(measurements)
        # TODO format that into engineering format
        return f"{mean:.3f} ± {std:.3f} {unit}"

    @classmethod
    def _compute_mean_and_std(cls, t):
        mean = float(t.mean())
        std = float(t.std(0, unbiased=t.numel() > 1))
        return mean, std


def no_split(benchmark, root):
    legacy_config = dict(benchmark.new_config)
    del legacy_config["split"]
    return legacy_config


def bool_split(name="train"):
    def legacy_config_map(benchmark, root):
        legacy_config = dict(benchmark.new_config)
        legacy_config[name] = legacy_config.pop("split") == "train"
        return legacy_config

    return legacy_config_map


def base_folder(rel_folder=None):
    if rel_folder is None:

        def rel_folder(benchmark):
            return benchmark.name

    elif not callable(rel_folder):
        name = rel_folder

        def rel_folder(_):
            return name

    def prepare_legacy_root(benchmark, root):
        files = list(root.glob("*"))
        folder = root / rel_folder(benchmark)
        folder.mkdir(parents=True)
        for file in files:
            shutil.move(str(file), str(folder))

        return folder

    return prepare_legacy_root


class JointTransform:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs, collections.abc.Sequence):
            inputs = inputs[0]

        if len(inputs) != len(self.transforms):
            raise RuntimeError(
                f"The number of inputs and transforms mismatches: {len(inputs)} != {len(self.transforms)}."
            )

        return tuple(transform(input) for transform, input in zip(self.transforms, inputs))


def caltech101_legacy_config_map(benchmark, root):
    legacy_config = no_split(benchmark, root)
    # The new dataset always returns the category and annotation
    legacy_config["target_type"] = ("category", "annotation")
    return legacy_config


mnist_base_folder = base_folder(lambda benchmark: pathlib.Path(benchmark.legacy_cls.__name__) / "raw")


def mnist_legacy_config_map(benchmark, root):
    return dict(train=benchmark.new_config.split == "train")


def emnist_prepare_legacy_root(benchmark, root):
    folder = mnist_base_folder(benchmark, root)
    shutil.move(str(folder / "emnist-gzip.zip"), str(folder / "gzip.zip"))
    return folder


def emnist_legacy_config_map(benchmark, root):
    legacy_config = mnist_legacy_config_map(benchmark, root)
    legacy_config["split"] = benchmark.new_config.image_set.replace("_", "").lower()
    return legacy_config


def qmnist_legacy_config_map(benchmark, root):
    legacy_config = mnist_legacy_config_map(benchmark, root)
    legacy_config["what"] = benchmark.new_config.split
    # The new dataset always returns the full label
    legacy_config["compat"] = False
    return legacy_config


def coco_legacy_config_map(benchmark, root):
    images, _ = benchmark.new_raw_dataset.resources(benchmark.new_config)
    return dict(
        root=str(root / pathlib.Path(images.file_name).stem),
        annFile=str(
            root / "annotations" / f"{benchmark.variant}_{benchmark.new_config.split}{benchmark.new_config.year}.json"
        ),
    )


def coco_prepare_legacy_root(benchmark, root):
    images, annotations = benchmark.new_raw_dataset.resources(benchmark.new_config)
    extract_archive(str(root / images.file_name))
    extract_archive(str(root / annotations.file_name))


DATASET_BENCHMARKS = [
    DatasetBenchmark(
        "caltech101",
        legacy_config_map=caltech101_legacy_config_map,
        prepare_legacy_root=base_folder(),
        legacy_special_options_map=lambda config: dict(
            download=True,
            transform=PILToTensor(),
            target_transform=JointTransform(torch.tensor, torch.tensor),
        ),
    ),
    DatasetBenchmark(
        "caltech256",
        legacy_config_map=no_split,
        prepare_legacy_root=base_folder(),
    ),
    DatasetBenchmark(
        "celeba",
        prepare_legacy_root=base_folder(),
        legacy_config_map=lambda benchmark: dict(
            split="valid" if benchmark.new_config.split == "val" else benchmark.new_config.split,
            # The new dataset always returns all annotations
            target_type=("attr", "identity", "bbox", "landmarks"),
        ),
    ),
    DatasetBenchmark(
        "cifar10",
        legacy_config_map=bool_split(),
    ),
    DatasetBenchmark(
        "cifar100",
        legacy_config_map=bool_split(),
    ),
    DatasetBenchmark(
        "emnist",
        prepare_legacy_root=emnist_prepare_legacy_root,
        legacy_config_map=emnist_legacy_config_map,
    ),
    DatasetBenchmark(
        "fashionmnist",
        prepare_legacy_root=mnist_base_folder,
        legacy_config_map=mnist_legacy_config_map,
    ),
    DatasetBenchmark(
        "kmnist",
        prepare_legacy_root=mnist_base_folder,
        legacy_config_map=mnist_legacy_config_map,
    ),
    DatasetBenchmark(
        "mnist",
        prepare_legacy_root=mnist_base_folder,
        legacy_config_map=mnist_legacy_config_map,
    ),
    DatasetBenchmark(
        "qmnist",
        prepare_legacy_root=mnist_base_folder,
        legacy_config_map=mnist_legacy_config_map,
    ),
    DatasetBenchmark(
        "sbd",
        legacy_cls=legacy_datasets.SBDataset,
        legacy_config_map=lambda benchmark: dict(
            image_set=benchmark.new_config.split,
            mode="boundaries" if benchmark.new_config.boundaries else "segmentation",
        ),
        legacy_special_options_map=lambda benchmark: dict(
            download=True,
            transforms=JointTransform(
                PILToTensor(), torch.tensor if benchmark.new_config.boundaries else PILToTensor()
            ),
        ),
    ),
    DatasetBenchmark("voc", legacy_cls=legacy_datasets.VOCDetection),
    DatasetBenchmark("imagenet", legacy_cls=legacy_datasets.ImageNet),
    DatasetBenchmark(
        "coco",
        variant="instances",
        legacy_cls=legacy_datasets.CocoDetection,
        new_config=dict(split="train", annotations="instances"),
        legacy_config_map=coco_legacy_config_map,
        prepare_legacy_root=coco_prepare_legacy_root,
        legacy_special_options_map=lambda benchmark: dict(transform=PILToTensor(), target_transform=None),
    ),
    DatasetBenchmark(
        "coco",
        variant="captions",
        legacy_cls=legacy_datasets.CocoCaptions,
        new_config=dict(split="train", annotations="captions"),
        legacy_config_map=coco_legacy_config_map,
        prepare_legacy_root=coco_prepare_legacy_root,
        legacy_special_options_map=lambda benchmark: dict(transform=PILToTensor(), target_transform=None),
    ),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="torchvision.prototype.datasets.benchmark.py",
        description="Utility to benchmark new datasets against their legacy variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("name", help="Name of the dataset to benchmark.")
    parser.add_argument(
        "--variant", help="Variant of the dataset. If omitted all available variants will be benchmarked."
    )

    parser.add_argument(
        "-n",
        "--num-starts",
        type=int,
        default=3,
        help="Number of warm and cold starts of each benchmark. Default to 3.",
    )
    parser.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=10_000,
        help="Maximum number of samples to draw during iteration benchmarks. Defaults to 10_000.",
    )

    parser.add_argument(
        "--nl",
        "--no-legacy",
        dest="legacy",
        action="store_false",
        help="Skip legacy benchmarks.",
    )
    parser.add_argument(
        "--nn",
        "--no-new",
        dest="new",
        action="store_false",
        help="Skip new benchmarks.",
    )
    parser.add_argument(
        "--ns",
        "--no-start",
        dest="start",
        action="store_false",
        help="Skip start benchmarks.",
    )
    parser.add_argument(
        "--ni",
        "--no-iteration",
        dest="iteration",
        action="store_false",
        help="Skip iteration benchmarks.",
    )

    parser.add_argument(
        "-t",
        "--temp-root",
        type=pathlib.Path,
        help=(
            "Root of the temporary legacy root directories. Use this if your system default temporary directory is on "
            "another storage device as the raw data to avoid distortions due to differing I/O stats."
        ),
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses used to load the data. Setting this to 0 (default) will load all data in the main "
            "process and thus disable multi-processing."
        ),
    )

    return parser.parse_args(argv or sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    try:
        main(
            args.name,
            variant=args.variant,
            legacy=args.legacy,
            new=args.new,
            start=args.start,
            iteration=args.iteration,
            num_starts=args.num_starts,
            num_samples=args.num_samples,
            temp_root=args.temp_root,
            num_workers=args.num_workers,
        )
    except Exception as error:
        msg = str(error)
        print(msg or f"Unspecified {type(error)} was raised during execution.", file=sys.stderr)
        sys.exit(1)
