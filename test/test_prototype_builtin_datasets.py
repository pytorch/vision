import io

import builtin_dataset_mocks
import pytest
from torch.utils.data.graph import traverse
from torchdata.datapipes.iter import IterDataPipe
from torchvision.prototype import datasets, features
from torchvision.prototype.datasets._api import DEFAULT_DECODER
from torchvision.prototype.utils._internal import sequence_to_str


def to_bytes(file):
    try:
        return file.read()
    finally:
        file.close()


def dataset_parametrization(*names, decoder=to_bytes):
    if not names:
        # TODO: Replace this with torchvision.prototype.datasets.list() as soon as all builtin datasets are supported
        names = (
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
            "sintel",
        )

    params = []
    for name in names:
        for config in datasets.info(name)._configs:
            id = f"{name}-{'-'.join([str(value) for value in config.values()])}"
            dataset, mock_info = builtin_dataset_mocks.load(name, decoder=decoder, **config)
            params.append(pytest.param(dataset, mock_info, id=id))

    return pytest.mark.parametrize(("dataset", "mock_info"), params)


class TestCommon:
    @dataset_parametrization()
    def test_smoke(self, dataset, mock_info):
        if not isinstance(dataset, IterDataPipe):
            raise AssertionError(f"Loading the dataset should return an IterDataPipe, but got {type(dataset)} instead.")

    @dataset_parametrization()
    def test_sample(self, dataset, mock_info):
        try:
            sample = next(iter(dataset))
        except Exception as error:
            raise AssertionError("Drawing a sample raised the error above.") from error

        if not isinstance(sample, dict):
            raise AssertionError(f"Samples should be dictionaries, but got {type(sample)} instead.")

        if not sample:
            raise AssertionError("Sample dictionary is empty.")

    @dataset_parametrization()
    def test_num_samples(self, dataset, mock_info):
        num_samples = 0
        for _ in dataset:
            num_samples += 1

        assert num_samples == mock_info["num_samples"]

    @dataset_parametrization()
    def test_decoding(self, dataset, mock_info):
        undecoded_features = {key for key, value in next(iter(dataset)).items() if isinstance(value, io.IOBase)}
        if undecoded_features:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(undecoded_features), separate_last='and ')} were not decoded."
            )

    @dataset_parametrization(decoder=DEFAULT_DECODER)
    def test_at_least_one_feature(self, dataset, mock_info):
        sample = next(iter(dataset))
        if not any(isinstance(value, features.Feature) for value in sample.values()):
            raise AssertionError("The sample contained no feature.")

    @dataset_parametrization()
    def test_traversable(self, dataset, mock_info):
        traverse(dataset)


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
