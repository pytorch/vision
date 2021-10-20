import pytest
from torchdata.datapipes.iter import IterableWrapper
from torchvision.prototype import datasets


@pytest.fixture
def make_minimal_dataset_info():
    def make(name="name", type=datasets.utils.DatasetType.RAW, categories=None, **kwargs):
        return datasets.utils.DatasetInfo(name, type=type, categories=categories or [], **kwargs)

    return make


@pytest.fixture
def patch_datasets(monkeypatch):
    datasets_storage = {}
    monkeypatch.setattr(datasets._api, "DATASETS", datasets_storage)
    return datasets_storage


@pytest.fixture
def dataset(mocker, make_minimal_dataset_info):
    info = make_minimal_dataset_info(valid_options=dict(split=("train", "test"), foo=("bar", "baz")))

    class DatasetMock(datasets.utils.Dataset):
        @property
        def info(self):
            return info

        def resources(self, config):
            return []

        def _make_datapipe(self, resource_dps, *, config, decoder):
            return IterableWrapper(resource_dps)

        to_datapipe = mocker.Mock()

    return DatasetMock()


class TestDatasetConfig:
    @staticmethod
    @pytest.fixture
    def options():
        return dict(foo="bar", baz=1)

    def test_creation_from_dict(self, options):
        datasets.utils.DatasetConfig(options)

    def test_creation_from_kwargs(self, options):
        datasets.utils.DatasetConfig(**options)

    def test_creation_mixed(self, options):
        datasets.utils.DatasetConfig(options, baz=options.pop("baz"))

    def test_creation_unhashable(self, options):
        with pytest.raises(TypeError):
            datasets.utils.DatasetConfig(options=options)

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
        assert set(datasets.utils.DatasetConfig(options).keys()) == set(options.keys())

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
    def info(make_minimal_dataset_info, valid_options):
        return make_minimal_dataset_info(valid_options=valid_options)

    def test_no_valid_options(self, make_minimal_dataset_info):
        info = make_minimal_dataset_info()
        assert info.default_config.split == "train"

    def test_valid_options_no_split(self, make_minimal_dataset_info):
        info = make_minimal_dataset_info(valid_options=dict(option=("argument",)))
        assert info.default_config.split == "train"

    def test_valid_options_no_train(self, make_minimal_dataset_info):
        with pytest.raises(ValueError):
            make_minimal_dataset_info(valid_options=dict(split=("test",)))

    def test_default_config(self, make_minimal_dataset_info, valid_options):
        default_config = datasets.utils.DatasetConfig({key: values[0] for key, values in valid_options.items()})

        assert make_minimal_dataset_info(valid_options=valid_options).default_config == default_config

    def test_make_config_unknown_option(self, info):
        with pytest.raises(ValueError):
            info.make_config(unknown_option=None)

    def test_make_config_invalid_argument(self, info):
        with pytest.raises(ValueError):
            info.make_config(split="unknown_split")

    def test_repr(self, make_minimal_dataset_info, valid_options):
        output = repr(make_minimal_dataset_info(valid_options=valid_options))

        assert isinstance(output, str)
        assert "DatasetInfo" in output
        for key, value in valid_options.items():
            assert f"{key}={str(value)[1:-1]}" in output

    @pytest.mark.parametrize("optional_info", ("citation", "homepage", "license"))
    def test_repr_optional_info(self, make_minimal_dataset_info, optional_info):
        sentinel = "sentinel"
        info = make_minimal_dataset_info(**{optional_info: sentinel})

        assert f"{optional_info}={sentinel}" in repr(info)


class TestDataset:
    @staticmethod
    @pytest.fixture
    def make_dataset(mocker):
        def make(name="name", valid_options=None, resources=None):
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
                    resources=mocker.Mock(return_value=[]) if resources is None else lambda self, config: resources,
                    _make_datapipe=mocker.Mock(),
                ),
            )
            return cls()

        return make

    def test_name(self, make_dataset):
        name = "sentinel"
        dataset = make_dataset(name=name)

        assert dataset.name == name

    def test_default_config(self, make_dataset):
        sentinel = "sentinel"
        valid_options = dict(split=(sentinel, "train"))
        dataset = make_dataset(valid_options=valid_options)

        assert dataset.default_config == datasets.utils.DatasetConfig(split=sentinel)

    def test_to_datapipe_config(self, make_dataset):
        dataset = make_dataset()
        config = datasets.utils.DatasetConfig(split="test")

        dataset.to_datapipe("", config=config)

        dataset.resources.assert_called_with(config)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["config"] == config

    def test_to_datapipe_default_config(self, make_dataset):
        dataset = make_dataset()
        config = dataset.default_config

        dataset.to_datapipe("")

        dataset.resources.assert_called_with(config)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["config"] == config

    def test_resources(self, mocker, make_dataset):
        resource_mock = mocker.Mock(spec=["to_datapipe"])
        sentinel = object()
        resource_mock.to_datapipe.return_value = sentinel
        dataset = make_dataset(resources=[resource_mock])

        root = "root"
        dataset.to_datapipe(root)

        resource_mock.to_datapipe.assert_called_with(root)

        (call_args, _) = dataset._make_datapipe.call_args
        assert call_args[0][0] is sentinel

    def test_decoder(self, make_dataset):
        dataset = make_dataset()

        sentinel = object()
        dataset.to_datapipe("", decoder=sentinel)

        (_, call_kwargs) = dataset._make_datapipe.call_args
        assert call_kwargs["decoder"] is sentinel
