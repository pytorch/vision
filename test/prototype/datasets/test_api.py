import pytest
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision.prototype import datasets
from torchvision.prototype.datasets import _api
from torchvision.prototype.datasets import _builtin
from torchvision.prototype.datasets.utils import Dataset, DatasetConfig


@pytest.fixture
def patch_datasets(monkeypatch):
    datasets_storage = {}
    monkeypatch.setattr(_api, "DATASETS", datasets_storage)
    return datasets_storage


@pytest.fixture
def dataset(mocker, make_minimal_dataset_info):
    info = make_minimal_dataset_info(
        valid_options=dict(split=("train", "test"), foo=("bar", "baz"))
    )

    class DatasetMock(Dataset):
        @property
        def info(self):
            return info

        def resources(self, config):
            return []

        def _make_datapipe(self, resource_dps, *, config, decoder):
            return IterableWrapper(resource_dps)

        to_datapipe = mocker.Mock()

    return DatasetMock()


def test_register(patch_datasets, dataset):
    datasets.register(dataset)
    assert patch_datasets == {dataset.name: dataset}


def test_list():
    builtin_dataset_names = {
        dataset_cls().name
        for name, dataset_cls in _builtin.__dict__.items()
        if not name.startswith("_")
        and isinstance(dataset_cls, type)
        and issubclass(dataset_cls, Dataset)
        and dataset_cls is not Dataset
    }
    assert set(datasets.list()) == builtin_dataset_names


class TestInfo:
    def test_main(self, patch_datasets, dataset):
        datasets.register(dataset)
        assert datasets.info(dataset.name) is dataset.info

    def test_unknown(self):
        with pytest.raises(ValueError):
            datasets.info("unknown")


class TestLoad:
    def test_main(self, patch_datasets, dataset):
        dp = object()
        to_datapipe_mock = dataset.to_datapipe
        to_datapipe_mock.return_value = dp

        options = dict(split="test", foo="baz")
        decoder = object()

        datasets.register(dataset)
        dp = datasets.load(dataset.name, decoder=decoder, **options)

        assert dp is dp

        root = datasets.home() / dataset.name
        config = DatasetConfig(options)

        to_datapipe_mock.assert_called_with(root, config=config, decoder=decoder)

    def test_unknown(self):
        with pytest.raises(ValueError):
            datasets.load("unknown")
