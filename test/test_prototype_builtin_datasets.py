import functools
import io

import builtin_dataset_mocks
import pytest
from torchdata.datapipes.iter import IterDataPipe
from torchvision.prototype import datasets
from torchvision.prototype.utils._internal import sequence_to_str


_loaders = []
_datasets = []

# TODO: this can be replaced by torchvision.prototype.datasets.list() as soon as all builtin datasets are supported
TMP = [
    "mnist",
    "fashionmnist",
    "kmnist",
    "emnist",
    "qmnist",
    "cifar10",
    "cifar100",
    "caltech256",
    "caltech101",
    "imagenet",
]
for name in TMP:
    loader = functools.partial(builtin_dataset_mocks.load, name)
    _loaders.append(pytest.param(loader, id=name))

    info = datasets.info(name)
    _datasets.extend(
        [
            pytest.param(*loader(**config), id=f"{name}-{'-'.join([str(value) for value in config.values()])}")
            for config in info._configs
        ]
    )

loaders = pytest.mark.parametrize("loader", _loaders)
builtin_datasets = pytest.mark.parametrize(("dataset", "mock_info"), _datasets)


class TestCommon:
    @builtin_datasets
    def test_smoke(self, dataset, mock_info):
        if not isinstance(dataset, IterDataPipe):
            raise AssertionError(f"Loading the dataset should return an IterDataPipe, but got {type(dataset)} instead.")

    @builtin_datasets
    def test_sample(self, dataset, mock_info):
        try:
            sample = next(iter(dataset))
        except Exception as error:
            raise AssertionError("Drawing a sample raised the error above.") from error

        if not isinstance(sample, dict):
            raise AssertionError(f"Samples should be dictionaries, but got {type(sample)} instead.")

        if not sample:
            raise AssertionError("Sample dictionary is empty.")

    @builtin_datasets
    def test_num_samples(self, dataset, mock_info):
        num_samples = 0
        for _ in dataset:
            num_samples += 1

        assert num_samples == mock_info["num_samples"]

    @builtin_datasets
    def test_decoding(self, dataset, mock_info):
        undecoded_features = {key for key, value in next(iter(dataset)).items() if isinstance(value, io.IOBase)}
        if undecoded_features:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(undecoded_features), separate_last='and ')} were not decoded."
            )


class TestQMNIST:
    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(builtin_dataset_mocks.load("qmnist", split=split)[0], id=split)
            for split in ("train", "test", "test10k", "test50k", "nist")
        ],
    )
    def test_extra_label(self, dataset):
        sample = next(iter(dataset))
        for key, type in (
            ("nist_hsf_series", int),
            ("nist_writer_id", int),
            ("digit_index", int),
            ("nist_label", int),
            ("global_digit_index", int),
            ("duplicate", bool),
            ("unused", bool),
        ):
            assert key in sample and isinstance(sample[key], type)
