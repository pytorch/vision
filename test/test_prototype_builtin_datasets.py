import functools
import io
import pickle
from pathlib import Path

import pytest
import torch
from builtin_dataset_mocks import DATASET_MOCKS, parametrize_dataset_mocks
from torch.testing._comparison import assert_equal, ObjectPair, TensorLikePair
from torch.utils.data import DataLoader
from torch.utils.data.graph import traverse
from torch.utils.data.graph_settings import get_all_graph_pipes
from torchdata.datapipes.iter import ShardingFilter, Shuffler
from torchvision._utils import sequence_to_str
from torchvision.prototype import datasets, transforms
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE
from torchvision.prototype.features import Image, Label

assert_samples_equal = functools.partial(
    assert_equal, pair_types=(TensorLikePair, ObjectPair), rtol=0, atol=0, equal_nan=True
)


def extract_datapipes(dp):
    return get_all_graph_pipes(traverse(dp, only_datapipe=True))


@pytest.fixture(autouse=True)
def test_home(mocker, tmp_path):
    mocker.patch("torchvision.prototype.datasets._api.home", return_value=str(tmp_path))
    mocker.patch("torchvision.prototype.datasets.home", return_value=str(tmp_path))
    yield tmp_path


def test_coverage():
    untested_datasets = set(datasets.list_datasets()) - DATASET_MOCKS.keys()
    if untested_datasets:
        raise AssertionError(
            f"The dataset(s) {sequence_to_str(sorted(untested_datasets), separate_last='and ')} "
            f"are exposed through `torchvision.prototype.datasets.load()`, but are not tested. "
            f"Please add mock data to `test/builtin_dataset_mocks.py`."
        )


@pytest.mark.filterwarnings("error")
class TestCommon:
    @pytest.mark.parametrize("name", datasets.list_datasets())
    def test_info(self, name):
        try:
            info = datasets.info(name)
        except ValueError:
            raise AssertionError("No info available.") from None

        if not (isinstance(info, dict) and all(isinstance(key, str) for key in info.keys())):
            raise AssertionError("Info should be a dictionary with string keys.")

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_smoke(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        if not isinstance(dataset, datasets.utils.Dataset):
            raise AssertionError(f"Loading the dataset should return an Dataset, but got {type(dataset)} instead.")

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_sample(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        try:
            sample = next(iter(dataset))
        except StopIteration:
            raise AssertionError("Unable to draw any sample.") from None
        except Exception as error:
            raise AssertionError("Drawing a sample raised the error above.") from error

        if not isinstance(sample, dict):
            raise AssertionError(f"Samples should be dictionaries, but got {type(sample)} instead.")

        if not sample:
            raise AssertionError("Sample dictionary is empty.")

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_num_samples(self, dataset_mock, config):
        dataset, mock_info = dataset_mock.load(config)

        assert len(list(dataset)) == mock_info["num_samples"]

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_no_vanilla_tensors(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        vanilla_tensors = {key for key, value in next(iter(dataset)).items() if type(value) is torch.Tensor}
        if vanilla_tensors:
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(vanilla_tensors), separate_last='and ')} contained vanilla tensors."
            )

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_transformable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        next(iter(dataset.map(transforms.Identity())))

    @pytest.mark.parametrize("only_datapipe", [False, True])
    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_traversable(self, dataset_mock, config, only_datapipe):
        dataset, _ = dataset_mock.load(config)

        traverse(dataset, only_datapipe=only_datapipe)

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_serializable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        pickle.dumps(dataset)

    # This has to be a proper function, since lambda's or local functions
    # cannot be pickled, but this is a requirement for the DataLoader with
    # multiprocessing, i.e. num_workers > 0
    def _collate_fn(self, batch):
        return batch

    @pytest.mark.parametrize("num_workers", [0, 1])
    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_data_loader(self, dataset_mock, config, num_workers):
        dataset, _ = dataset_mock.load(config)

        dl = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

        next(iter(dl))

    # TODO: we need to enforce not only that both a Shuffler and a ShardingFilter are part of the datapipe, but also
    #  that the Shuffler comes before the ShardingFilter. Early commits in https://github.com/pytorch/vision/pull/5680
    #  contain a custom test for that, but we opted to wait for a potential solution / test from torchdata for now.
    @parametrize_dataset_mocks(DATASET_MOCKS)
    @pytest.mark.parametrize("annotation_dp_type", (Shuffler, ShardingFilter))
    def test_has_annotations(self, dataset_mock, config, annotation_dp_type):
        dataset, _ = dataset_mock.load(config)

        if not any(isinstance(dp, annotation_dp_type) for dp in extract_datapipes(dataset)):
            raise AssertionError(f"The dataset doesn't contain a {annotation_dp_type.__name__}() datapipe.")

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_save_load(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        sample = next(iter(dataset))

        with io.BytesIO() as buffer:
            torch.save(sample, buffer)
            buffer.seek(0)
            assert_samples_equal(torch.load(buffer), sample)

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_infinite_buffer_size(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        for dp in extract_datapipes(dataset):
            if hasattr(dp, "buffer_size"):
                # TODO: replace this with the proper sentinel as soon as https://github.com/pytorch/data/issues/335 is
                #  resolved
                assert dp.buffer_size == INFINITE_BUFFER_SIZE

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_has_length(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        assert len(dataset) > 0


@parametrize_dataset_mocks(DATASET_MOCKS["qmnist"])
class TestQMNIST:
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


@parametrize_dataset_mocks(DATASET_MOCKS["gtsrb"])
class TestGTSRB:
    def test_label_matches_path(self, dataset_mock, config):
        # We read the labels from the csv files instead. But for the trainset, the labels are also part of the path.
        # This test makes sure that they're both the same
        if config["split"] != "train":
            return

        dataset, _ = dataset_mock.load(config)

        for sample in dataset:
            label_from_path = int(Path(sample["path"]).parent.name)
            assert sample["label"] == label_from_path


@parametrize_dataset_mocks(DATASET_MOCKS["usps"])
class TestUSPS:
    def test_sample_content(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        for sample in dataset:
            assert "image" in sample
            assert "label" in sample

            assert isinstance(sample["image"], Image)
            assert isinstance(sample["label"], Label)

            assert sample["image"].shape == (1, 16, 16)
