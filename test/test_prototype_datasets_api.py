import unittest.mock

import pytest
from torchvision.prototype import datasets


def make_minimal_dataset_info(name="name", categories=None, **kwargs):
    # TODO: remove this?
    return dict(categories=categories or [], **kwargs)


class TestDataset:
    class DatasetMock(datasets.utils.Dataset):
        def __init__(self, info=None, *, resources=None):
            self._info = info or make_minimal_dataset_info(valid_options=dict(split=("train", "test")))
            self.resources = unittest.mock.Mock(return_value=[]) if resources is None else lambda config: resources
            self._make_datapipe = unittest.mock.Mock()
            super().__init__()

        def _make_info(self):
            return self._info

        def resources(self, config):
            # This method is just defined to appease the ABC, but will be overwritten at instantiation
            pass

        def _make_datapipe(self, resource_dps, *, config):
            # This method is just defined to appease the ABC, but will be overwritten at instantiation
            pass

    def test_name(self):
        name = "sentinel"
        dataset = self.DatasetMock(make_minimal_dataset_info(name=name))

        assert dataset.name == name

    def test_default_config(self):
        sentinel = "sentinel"
        dataset = self.DatasetMock(info=make_minimal_dataset_info(valid_options=dict(split=(sentinel, "train"))))

        assert dataset.default_config == datasets.utils.DatasetConfig(split=sentinel)

    @pytest.mark.parametrize(
        ("config", "kwarg"),
        [
            pytest.param(*(datasets.utils.DatasetConfig(split="test"),) * 2, id="specific"),
            pytest.param(DatasetMock().default_config, None, id="default"),
        ],
    )
    def test_load_config(self, config, kwarg):
        dataset = self.DatasetMock()

        dataset.load("", config=kwarg)

        dataset.resources.assert_called_with(config)

        _, call_kwargs = dataset._make_datapipe.call_args
        assert call_kwargs["config"] == config

    def test_missing_dependencies(self):
        dependency = "fake_dependency"
        dataset = self.DatasetMock(make_minimal_dataset_info(dependencies=(dependency,)))
        with pytest.raises(ModuleNotFoundError, match=dependency):
            dataset.load("root")

    def test_resources(self, mocker):
        resource_mock = mocker.Mock(spec=["load"])
        sentinel = object()
        resource_mock.load.return_value = sentinel
        dataset = self.DatasetMock(resources=[resource_mock])

        root = "root"
        dataset.load(root)

        (call_args, _) = resource_mock.load.call_args
        assert call_args[0] == root

        (call_args, _) = dataset._make_datapipe.call_args
        assert call_args[0][0] is sentinel
