import argparse
import contextlib
import copy
import inspect
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
    try:
        benchmark = next(benchmark for benchmark in DATASET_BENCHMARKS if benchmark.name == name)
    except StopIteration as error:
        raise ValueError(f"No DatasetBenchmark available for dataset '{name}'") from error

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

    @contextlib.contextmanager
    def legacy_root(self):
        new_root = new_datasets.home() / self.name
        legacy_root = pathlib.Path(tempfile.mkdtemp())

        for resource in self.new_raw_dataset.resources(self.new_raw_dataset.default_config):
            (legacy_root / resource.file_name).symlink_to(new_root / resource.file_name)

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
        available_kwargs = {name for name in inspect.signature(self.legacy_cls.__init__).parameters}
        available_special_kwargs = self._SPECIAL_KWARGS.intersection(available_kwargs)

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


def base_folder(rel_folder=None):
    def prepare_legacy_root(benchmark: DatasetBenchmark, root: pathlib.Path):
        files = root.glob("*")
        folder = root / (rel_folder or benchmark.name)
        folder.mkdir()
        for file in files:
            shutil.move(str(file), str(folder))

    return prepare_legacy_root


DATASET_BENCHMARKS = [
    DatasetBenchmark("caltech101", prepare_legacy_root=base_folder()),
    DatasetBenchmark("caltech256", prepare_legacy_root=base_folder()),
    DatasetBenchmark(
        "celeba",
        prepare_legacy_root=base_folder(),
        legacy_config_map=lambda config: dict(
            split="valid" if config.split == "val" else config.split,
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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str)
    parser.add_argument("--number", "-n", type=int, default=5, help="Number of iterations of each benchmark")

    return parser.parse_args(args or sys.argv[1:])


if __name__ == "__main__":
    args = parse_args(["cifar10"])

    try:
        main(args.name, number=args.number)
    except Exception as error:
        msg = str(error)
        print(msg or f"Unspecified {type(error)} was raised during execution.", file=sys.stderr)
        status = 1
    else:
        status = 0
    sys.exit(status)
