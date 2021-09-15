import functools
import io
import itertools
from collections import OrderedDict

import pytest

import torch
from torch.utils.data import IterDataPipe

import torchvision.datasets
from torchvision.prototype.datasets.decoder import pil

import dataset_mocks


_loaders = []
_datasets = []
for name in torchvision.prototype.datasets.list():
    loader = functools.partial(dataset_mocks.load, name)
    _loaders.append(loader)

    info = torchvision.prototype.datasets.info(name)
    for combination in itertools.product(*info.options.values()):
        options = OrderedDict(sorted(zip(info.options.keys(), combination)))
        _datasets.append(
            pytest.param(
                *loader(**options),
                id=f"{name}-{'-'.join([str(value) for value in options.values()])}",
            )
        )
loaders = pytest.mark.parametrize("loader", _loaders)
datasets = pytest.mark.parametrize(("dataset", "mock_info"), _datasets)


class TestCommon:
    @datasets
    def test_smoke(self, dataset, mock_info):
        if not isinstance(dataset, IterDataPipe):
            raise AssertionError("FIXME")

    @datasets
    def test_sample(self, dataset, mock_info):
        try:
            sample = next(iter(dataset))
        except Exception as error:
            raise AssertionError("FIXME") from error

        if not isinstance(sample, dict):
            raise AssertionError("FIXME")

        if not sample:
            raise AssertionError("FIXME")

    @datasets
    def test_num_samples(self, dataset, mock_info):
        num_samples = 0
        for _ in dataset:
            num_samples += 1

        assert num_samples == mock_info["num_samples"]

    @loaders
    def test_decoding(self, loader):
        dataset, _ = loader(decoder=pil)
        undecoded_features = {key for key, value in next(iter(dataset)).items() if isinstance(value, io.IOBase)}
        if undecoded_features:
            raise AssertionError(f"The values of key(s) {', '.join(sorted(undecoded_features))} were not decoded.")


class TestQMNIST:
    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(dataset_mocks.load("qmnist", split=split)[0], id=split)
            for split in ("train", "test", "test10k", "test50k", "nist")
        ],
    )
    def test_extra_label(self, dataset):
        sample = next(iter(dataset))
        for key, dtype in (
            ("nist_hsf_series", torch.int32),
            ("nist_writer_id", torch.int32),
            ("digit_index", torch.int32),
            ("nist_label", torch.int32),
            ("global_digit_index", torch.int32),
            ("duplicate", torch.bool),
            ("unused", torch.bool),
        ):
            assert key in sample
            value = sample[key]
            assert isinstance(value, torch.Tensor)
            assert value.dtype == dtype
