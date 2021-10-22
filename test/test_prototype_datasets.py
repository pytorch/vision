import unittest.mock

import pytest
from torchvision.prototype import datasets
from torchvision.prototype.datasets import _api


def make_minimal_dataset_info(name="name", type=datasets.utils.DatasetType.RAW, categories=None, **kwargs):
    return datasets.utils.DatasetInfo(name, type=type, categories=categories or [], **kwargs)


@pytest.fixture
def patch_datasets(mocker):
    registered_datasets = {}
    mocker.patch.dict(_api.DATASETS, registered_datasets, clear=True)
    return registered_datasets


class TestDatasetConfig:
    @pytest.fixture
    def options(self):
        return dict(foo="bar", baz=1)

    def test_creation_from_dict(self, options):
        datasets.utils.DatasetConfig(options)

    def test_creation_from_kwargs(self, options):
        datasets.utils.DatasetConfig(**options)

    def test_creation_mixed(self, options):
        datasets.utils.DatasetConfig(options, baz=options.pop("baz"))

    def test_creation_unhashable(self, options):
        with pytest.raises(TypeError):
            datasets.utils.DatasetConfig(foo=[])

    def test_getitem(self, options):
        config = datasets.utils.DatasetConfig(options)

        for key, value in options.items():
            assert config[key] == value

    def test_getitem_unknown(self):
        with pytest.raises(KeyError):
            datasets.utils.DatasetConfig()["unknown"]

    def test_iter(self, options):
        assert set(iter(datasets.utils.DatasetConfig(options))) == set(options.keys())

    def test_len(self, options):
        assert len(datasets.utils.DatasetConfig(options)) == len(options)

    def test_getattr(self, options):
        config = datasets.utils.DatasetConfig(options)

        for key, value in options.items():
            assert getattr(config, key) == value

    def test_getattr_unknown(self):
        with pytest.raises(AttributeError):
            datasets.utils.DatasetConfig().unknown

    def test_setitem(self):
        config = datasets.utils.DatasetConfig()

        with pytest.raises(RuntimeError):
            config["foo"] = "bar"

    def test_setattr(self):
        config = datasets.utils.DatasetConfig()

        with pytest.raises(RuntimeError):
            config.foo = "bar"

    def test_delitem(self, options):
        config = datasets.utils.DatasetConfig(options)

        with pytest.raises(RuntimeError):
            del config["foo"]

    def test_delattr(self, options):
        config = datasets.utils.DatasetConfig(options)

        with pytest.raises(RuntimeError):
            del config.foo

    def test_eq(self, options):
        config1 = datasets.utils.DatasetConfig(options)
        config2 = datasets.utils.DatasetConfig(options)

        assert config1 == config2

    def test_neq(self, options):
        config1 = datasets.utils.DatasetConfig(options)
        config2 = datasets.utils.DatasetConfig()

        assert config1 != config2

    def test_repr(self, options):
        output = repr(datasets.utils.DatasetConfig(options))

        assert isinstance(output, str)
        assert "DatasetConfig" in output
        for key, value in options.items():
            assert f"{key}={value}" in output

    def test_contains(self, options):
        config = datasets.utils.DatasetConfig(options)

        for key in options.keys():
            assert key in config

    def test_keys(self, options):
        assert datasets.utils.DatasetConfig(options).keys() == options.keys()

    def test_values(self, options):
        assert set(datasets.utils.DatasetConfig(options).values()) == set(options.values())

    def test_get(self, options):
        config = datasets.utils.DatasetConfig(options)

        for key, value in options.items():
            assert config.get(key) == value

    def test_get_default(self, options):
        sentinel = object()

        assert datasets.utils.DatasetConfig().get("unknown", sentinel) is sentinel

    def test_ne(self, options):
        config1 = datasets.utils.DatasetConfig(options)
        options["baz"] = 2
        config2 = datasets.utils.DatasetConfig(options)

        assert config1 != config2


class TestDatasetInfo:
    @staticmethod
    @pytest.fixture
    def valid_options():
        return dict(split=("train", "test"), foo=("bar", "baz"))

    @staticmethod
    @pytest.fixture
    def info(valid_options):
        return make_minimal_dataset_info(valid_options=valid_options)

    def test_no_valid_options(self):
        info = make_minimal_dataset_info()
        assert info.default_config.split == "train"

    def test_valid_options_no_split(self):
        info = make_minimal_dataset_info(valid_options=dict(option=("argument",)))
        assert info.default_config.split == "train"

    def test_valid_options_no_train(self):
        with pytest.raises(ValueError):
            make_minimal_dataset_info(valid_options=dict(split=("test",)))

    def test_default_config(self, valid_options):
        default_config = datasets.utils.DatasetConfig({key: values[0] for key, values in valid_options.items()})

        assert make_minimal_dataset_info(valid_options=valid_options).default_config == default_config

    def test_make_config_unknown_option(self, info):
        with pytest.raises(ValueError):
            info.make_config(unknown_option=None)

    def test_make_config_invalid_argument(self, info):
        with pytest.raises(ValueError):
            info.make_config(split="unknown_split")

    def test_repr(self, valid_options):
        output = repr(make_minimal_dataset_info(valid_options=valid_options))

        assert isinstance(output, str)
        assert "DatasetInfo" in output
        for key, value in valid_options.items():
            assert f"{key}={str(value)[1:-1]}" in output

    @pytest.mark.parametrize("optional_info", ("citation", "homepage", "license"))
    def test_repr_optional_info(self, optional_info):
        sentinel = "sentinel"
        info = make_minimal_dataset_info(**{optional_info: sentinel})

        assert f"{optional_info}={sentinel}" in repr(info)


class TestDataset:
    def make_dataset_mock(self, name="name", valid_options=None, resources=None):
        cls = type(
            "DatasetMock",
            (datasets.utils.Dataset,),
            dict(
                info=datasets.utils.DatasetInfo(
                    name,
                    type=datasets.utils.DatasetType.RAW,
                    categories=[],
                    valid_options=valid_options or dict(split=("train", "test")),
                ),
                resources=unittest.mock.Mock(return_value=[]) if resources is None else lambda self, config: resources,
                _make_datapipe=unittest.mock.Mock(),
            ),
        )
        return cls()

    def test_name(self):
        name = "sentinel"
        dataset = self.make_dataset_mock(name=name)

        assert dataset.name == name

    def test_default_config(self):
        sentinel = "sentinel"
        valid_options = dict(split=(sentinel, "train"))
        dataset = self.make_dataset_mock(valid_options=valid_options)

        assert dataset.default_config == datasets.utils.DatasetConfig(split=sentinel)

    def test_to_datapipe_config(self):
        dataset = self.make_dataset_mock()
        config = datasets.utils.DatasetConfig(split="test")

        dataset.to_datapipe("", config=config)

        dataset.resources.assert_called_with(config)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["config"] == config

    def test_to_datapipe_default_config(self):
        dataset = self.make_dataset_mock()
        config = dataset.default_config

        dataset.to_datapipe("")

        dataset.resources.assert_called_with(config)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["config"] == config

    def test_resources(self, mocker):
        resource_mock = mocker.Mock(spec=["to_datapipe"])
        sentinel = object()
        resource_mock.to_datapipe.return_value = sentinel
        dataset = self.make_dataset_mock(resources=[resource_mock])

        root = "root"
        dataset.to_datapipe(root)

        resource_mock.to_datapipe.assert_called_with(root)

        (call_args, _) = dataset._make_datapipe.call_args
        assert call_args[0][0] is sentinel

    def test_decoder(self):
        dataset = self.make_dataset_mock()

        sentinel = object()
        dataset.to_datapipe("", decoder=sentinel)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["decoder"] is sentinel
