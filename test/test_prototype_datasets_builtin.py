import io
import pickle
from collections import deque
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as transforms

from builtin_dataset_mocks import DATASET_MOCKS, parametrize_dataset_mocks
from torch.testing._comparison import not_close_error_metas, ObjectPair, TensorLikePair

# TODO: replace with torchdata.dataloader2.DataLoader2 as soon as it is stable-ish
from torch.utils.data import DataLoader

# TODO: replace with torchdata equivalent as soon as it is available
from torch.utils.data.graph_settings import get_all_graph_pipes

from torchdata.dataloader2.graph.utils import traverse_dps
from torchdata.datapipes.iter import ShardingFilter, Shuffler
from torchdata.datapipes.utils import StreamWrapper
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.prototype import datasets
from torchvision.prototype.datasets.utils import EncodedImage
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE
from torchvision.prototype.tv_tensors import Label
from torchvision.transforms.v2._utils import is_pure_tensor


def assert_samples_equal(*args, msg=None, **kwargs):
    error_metas = not_close_error_metas(
        *args, pair_types=(TensorLikePair, ObjectPair), rtol=0, atol=0, equal_nan=True, **kwargs
    )
    if error_metas:
        raise error_metas[0].to_error(msg)


def extract_datapipes(dp):
    return get_all_graph_pipes(traverse_dps(dp))


def consume(iterator):
    # Copied from the official itertools recipes: https://docs.python.org/3/library/itertools.html#itertools-recipes
    deque(iterator, maxlen=0)


def next_consume(iterator):
    item = next(iterator)
    consume(iterator)
    return item


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
            sample = next_consume(iter(dataset))
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

    @pytest.fixture
    def log_session_streams(self):
        debug_unclosed_streams = StreamWrapper.debug_unclosed_streams
        try:
            StreamWrapper.debug_unclosed_streams = True
            yield
        finally:
            StreamWrapper.debug_unclosed_streams = debug_unclosed_streams

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_stream_closing(self, log_session_streams, dataset_mock, config):
        def make_msg_and_close(head):
            unclosed_streams = []
            for stream in list(StreamWrapper.session_streams.keys()):
                unclosed_streams.append(repr(stream.file_obj))
                stream.close()
            unclosed_streams = "\n".join(unclosed_streams)
            return f"{head}\n\n{unclosed_streams}"

        if StreamWrapper.session_streams:
            raise pytest.UsageError(make_msg_and_close("A previous test did not close the following streams:"))

        dataset, _ = dataset_mock.load(config)

        consume(iter(dataset))

        if StreamWrapper.session_streams:
            raise AssertionError(make_msg_and_close("The following streams were not closed after a full iteration:"))

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_no_unaccompanied_pure_tensors(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)
        sample = next_consume(iter(dataset))

        pure_tensors = {key for key, value in sample.items() if is_pure_tensor(value)}

        if pure_tensors and not any(
            isinstance(item, (tv_tensors.Image, tv_tensors.Video, EncodedImage)) for item in sample.values()
        ):
            raise AssertionError(
                f"The values of key(s) "
                f"{sequence_to_str(sorted(pure_tensors), separate_last='and ')} contained pure tensors, "
                f"but didn't find any (encoded) image or video."
            )

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_transformable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        dataset = dataset.map(transforms.Identity())

        consume(iter(dataset))

    @parametrize_dataset_mocks(DATASET_MOCKS)
    def test_traversable(self, dataset_mock, config):
        dataset, _ = dataset_mock.load(config)

        traverse_dps(dataset)

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

        consume(dl)

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

        sample = next_consume(iter(dataset))

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

        sample = next_consume(iter(dataset))
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

            assert isinstance(sample["image"], tv_tensors.Image)
            assert isinstance(sample["label"], Label)

            assert sample["image"].shape == (1, 16, 16)
