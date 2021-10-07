import argparse
import contextlib
import copy
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

import torch
from torchvision import datasets as legacy_datasets
from torchvision.datasets.vision import StandardTransform
from torchvision.prototype import datasets as new_datasets
from torchvision.transforms import ToTensor


def main(name, *, number):
    for benchmark in DATASET_BENCHMARKS:
        if benchmark.name == name:
            break
    else:
        raise ValueError(f"No DatasetBenchmark available for dataset '{name}'")

    print("legacy", "cold_start", Measurement.time(benchmark.legacy_cold_start, number=number))
    print("legacy", "warm_start", Measurement.time(benchmark.legacy_warm_start, number=number))
    print("legacy", "iter", Measurement.iterations_per_time(benchmark.legacy_iteration, number=number))

    print("new", "cold_start", Measurement.time(benchmark.new_cold_start, number=number))
    print("new", "iter", Measurement.iterations_per_time(benchmark.new_iter, number=number))


class DatasetBenchmark:
    def __init__(
        self,
        name: str,
        *,
        legacy_cls=None,
        new_config=None,
        legacy_config_map=None,
        legacy_special_options_map=None,
        prepare_legacy_root=None,
    ):
        self.name = name

        self.new_raw_dataset = new_datasets._api.find(name)
        self.legacy_cls = legacy_cls or self._find_legacy_cls()

        if new_config is None:
            new_config = self.new_raw_dataset.default_config
        elif isinstance(new_config, dict):
            new_config = new_datasets.utils.DatasetConfig(new_config)
        self.new_config = new_config
        self.legacy_config = (legacy_config_map or dict)(copy.copy(new_config))

        self.legacy_special_options = (legacy_special_options_map or self._legacy_special_options_map)(
            copy.copy(new_config)
        )

        self.prepare_legacy_root = prepare_legacy_root

    def new_dataset(self):
        return new_datasets.load(self.name, **self.new_config)

    def new_cold_start(self, timer):
        with timer:
            dataset = self.new_dataset()
            next(iter(dataset))

    def new_iter(self, timer):
        dataset = self.new_dataset()
        num_samples = 0

        with timer:
            for _ in dataset:
                num_samples += 1

        return num_samples

    def suppress_output(self):
        @contextlib.contextmanager
        def context_manager():
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        return context_manager()

    def legacy_dataset(self, root, *, download=None):
        special_options = self.legacy_special_options.copy()
        if "download" in special_options and download is not None:
            special_options["download"] = download
        with self.suppress_output():
            return self.legacy_cls(str(root), **self.legacy_config, **special_options)

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
    def legacy_root(self):
        new_root = new_datasets.home() / self.name
        legacy_root = pathlib.Path(tempfile.mkdtemp())

        for file_name in self._find_resource_file_names():
            (legacy_root / file_name).symlink_to(new_root / file_name)

        if self.prepare_legacy_root:
            self.prepare_legacy_root(self, legacy_root)

        with self.patch_download_and_integrity_checks():
            try:
                yield legacy_root
            finally:
                shutil.rmtree(legacy_root)

    def legacy_cold_start(self, timer):
        with self.legacy_root() as root:
            with timer:
                dataset = self.legacy_dataset(root)
                next(iter(dataset))

    def legacy_warm_start(self, timer):
        with self.legacy_root() as root:
            self.legacy_dataset(root)
            with timer:
                dataset = self.legacy_dataset(root, download=False)
                next(iter(dataset))

    def legacy_iteration(self, timer):
        with self.legacy_root() as root:
            dataset = self.legacy_dataset(root)
            with timer:
                for _ in dataset:
                    pass

        return len(dataset)

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

    def _legacy_special_options_map(self, config):
        available_parameters = set()

        for cls in self.legacy_cls.__mro__:
            if cls is legacy_datasets.VisionDataset:
                break

            available_parameters.update(inspect.signature(cls.__init__).parameters)

        available_special_kwargs = self._SPECIAL_KWARGS.intersection(available_parameters)

        special_options = dict()

        if "download" in available_special_kwargs:
            special_options["download"] = True

        if "transform" in available_special_kwargs:
            special_options["transform"] = ToTensor()
            if "target_transform" in available_special_kwargs:
                special_options["target_transform"] = torch.tensor
        elif "transforms" in available_special_kwargs:
            special_options["transforms"] = StandardTransform(ToTensor(), ToTensor())

        return special_options


class Measurement:
    @classmethod
    def time(cls, fn, *, number):
        results = Measurement._timeit(fn, number=number)

        times = torch.tensor(tuple(zip(*results))[1])

        mean, std = Measurement._compute_mean_and_std(times)
        # TODO format that into engineering format
        return f"{mean:.3g} ± {std:.3g} s"

    @classmethod
    def iterations_per_time(cls, fn, *, number):
        outputs, times = zip(*Measurement._timeit(fn, number=number))

        num_samples = outputs[0]
        assert all(other_num_samples == num_samples for other_num_samples in outputs[1:])
        iterations_per_time = torch.tensor(num_samples) / torch.tensor(times)

        mean, std = Measurement._compute_mean_and_std(iterations_per_time)
        # TODO format that into engineering format
        return f"{mean:.1f} ± {std:.1f} it/s"

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
    def _timeit(cls, fn, *, number):
        results = []
        for _ in range(number):
            timer = cls.Timer()
            output = fn(timer)
            results.append((output, timer.delta))
        return results

    @classmethod
    def _compute_mean_and_std(cls, t):
        t = t.flatten()
        mean = float(t.mean())
        std = float(t.std(0, unbiased=t.numel() > 1))
        return mean, std


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


mnist_base_folder = base_folder(lambda benchmark: pathlib.Path(benchmark.legacy_cls.__name__) / "raw")


def mnist_legacy_config_map(config):
    return dict(train=config.split == "train")


def emnist_prepare_legacy_root(benchmark, root):
    folder = mnist_base_folder(benchmark, root)
    shutil.move(str(folder / "emnist-gzip.zip"), str(folder / "gzip.zip"))
    return folder


def emnist_legacy_config_map(config):
    legacy_config = mnist_legacy_config_map(config)
    legacy_config["split"] = config.image_set.replace("_", "").lower()
    return legacy_config


def qmnist_legacy_config_map(config):
    legacy_config = mnist_legacy_config_map(config)
    legacy_config["what"] = config.split
    # The new dataset always returns the full label
    legacy_config["compat"] = False
    return legacy_config


DATASET_BENCHMARKS = [
    DatasetBenchmark("caltech101", prepare_legacy_root=base_folder()),
    DatasetBenchmark("caltech256", prepare_legacy_root=base_folder()),
    DatasetBenchmark(
        "celeba",
        prepare_legacy_root=base_folder(),
        legacy_config_map=lambda config: dict(
            split="valid" if config.split == "val" else config.split,
            # The new dataset always returns all annotations
            target_type=("attr", "identity", "bbox", "landmarks"),
        ),
    ),
    DatasetBenchmark(
        "cifar10",
        legacy_config_map=lambda config: dict(train=config.split == "train"),
    ),
    DatasetBenchmark(
        "cifar100",
        legacy_config_map=lambda config: dict(train=config.split == "train"),
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
        legacy_config_map=lambda config: dict(
            image_set=config.split,
            mode="boundaries" if config.boundaries else "segmentation",
        ),
        legacy_special_options_map=lambda config: dict(
            download=True,
            transforms=StandardTransform(ToTensor(), torch.tensor if config.boundaries else ToTensor()),
        ),
    ),
    DatasetBenchmark("voc", legacy_cls=legacy_datasets.VOCDetection),
]


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str)
    parser.add_argument("--number", "-n", type=int, default=5, help="Number of iterations of each benchmark")

    return parser.parse_args(args or sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    try:
        main(args.name, number=args.number)
    except Exception as error:
        msg = str(error)
        print(msg or f"Unspecified {type(error)} was raised during execution.", file=sys.stderr)
        sys.exit(1)
