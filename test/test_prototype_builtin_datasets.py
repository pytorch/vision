import io

import pytest
import torch
from builtin_dataset_mocks import parametrize_dataset_mocks, DATASET_MOCKS
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe as ShardingFilter
from torch.utils.data.graph import traverse
from torchdata.datapipes.iter import IterDataPipe, Shuffler
from torchvision.prototype import transforms
from torchvision.prototype.utils._internal import sequence_to_str


@parametrize_dataset_mocks(DATASET_MOCKS)
class TestCommon:
    def test_smoke(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)
        if not isinstance(dataset, IterDataPipe):
            raise AssertionError(f"Loading the dataset should return an IterDataPipe, but got {type(dataset)} instead.")

    def test_sample(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        try:
            sample = next(iter(dataset))
        except Exception as error:
            raise AssertionError("Drawing a sample raised the error above.") from error

        if not isinstance(sample, dict):
            raise AssertionError(f"Samples should be dictionaries, but got {type(sample)} instead.")

        if not sample:
            raise AssertionError("Sample dictionary is empty.")

    def test_num_samples(self, dataset_mock, config):
        dataset, mock_info = dataset_mock.load(config)

        num_samples = 0
        for _ in dataset:
            num_samples += 1

        assert num_samples == mock_info["num_samples"]

    def test_decoding(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        undecoded_features = {key for key, value in next(iter(dataset)).items() if isinstance(value, io.IOBase)}
        if undecoded_features:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(undecoded_features), separate_last='and ')} were not decoded."
            )

    def test_no_vanilla_tensors(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        vanilla_tensors = {key for key, value in next(iter(dataset)).items() if type(value) is torch.Tensor}
        if vanilla_tensors:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(vanilla_tensors), separate_last='and ')} contained vanilla tensors."
            )

    def test_transformable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        next(iter(dataset.map(transforms.Identity())))

    def test_traversable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        traverse(dataset)

    @pytest.mark.parametrize("annotation_dp_type", (Shuffler, ShardingFilter), ids=lambda type: type.__name__)
    def test_has_annotations(self, dataset_mock, config, annotation_dp_type):
        def scan(graph):
            for node, sub_graph in graph.items():
                yield node
                yield from scan(sub_graph)

        dataset, _ = dataset_mock.load(config)

        for dp in scan(traverse(dataset)):
            if type(dp) is annotation_dp_type:
                break
        else:
            raise AssertionError(f"The dataset doesn't comprise a {annotation_dp_type.__name__}() datapipe.")


class TestQMNIST:
    @parametrize_dataset_mocks([mock for mock in DATASET_MOCKS if mock.name == "qmnist"])
    def test_extra_label(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

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
