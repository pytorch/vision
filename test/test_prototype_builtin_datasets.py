import io

import builtin_dataset_mocks
import pytest
import torch
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe as ShardingFilter
from torch.utils.data.graph import traverse
from torchdata.datapipes.iter import IterDataPipe, Shuffler
from torchvision.prototype import datasets, transforms
from torchvision.prototype.datasets._api import DEFAULT_DECODER
from torchvision.prototype.utils._internal import sequence_to_str


def to_bytes(file):
    return file.read()


def config_id(name, config):
    parts = [name]
    for name, value in config.items():
        if isinstance(value, bool):
            part = ("" if value else "no_") + name
        else:
            part = str(value)
        parts.append(part)
    return "-".join(parts)


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
            "coco",
        )

    return pytest.mark.parametrize(
        ("dataset", "mock_info"),
        [
            pytest.param(*builtin_dataset_mocks.load(name, decoder=decoder, **config), id=config_id(name, config))
            for name in names
            for config in datasets.info(name)._configs
        ],
    )


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
    def test_no_vanilla_tensors(self, dataset, mock_info):
        vanilla_tensors = {key for key, value in next(iter(dataset)).items() if type(value) is torch.Tensor}
        if vanilla_tensors:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(vanilla_tensors), separate_last='and ')} contained vanilla tensors."
            )

    @dataset_parametrization()
    def test_transformable(self, dataset, mock_info):
        next(iter(dataset.map(transforms.Identity())))

    @dataset_parametrization()
    def test_traversable(self, dataset, mock_info):
        traverse(dataset)

    @dataset_parametrization()
    @pytest.mark.parametrize("annotation_dp_type", (Shuffler, ShardingFilter), ids=lambda type: type.__name__)
    def test_has_annotations(self, dataset, mock_info, annotation_dp_type):
        def scan(graph):
            for node, sub_graph in graph.items():
                yield node
                yield from scan(sub_graph)

        for dp in scan(traverse(dataset)):
            if type(dp) is annotation_dp_type:
                break
        else:
            raise AssertionError(f"The dataset doesn't comprise a {annotation_dp_type.__name__}() datapipe.")


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
