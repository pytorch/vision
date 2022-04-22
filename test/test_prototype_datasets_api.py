import unittest.mock

import pytest
from torchvision.prototype import datasets
from torchvision.prototype.utils._internal import FrozenMapping, FrozenBunch


def make_minimal_dataset_info(name="name", categories=None, **kwargs):
    return datasets.utils.DatasetInfo(name, categories=categories or [], **kwargs)


class TestFrozenMapping:
    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            pytest.param((dict(foo="bar", baz=1),), dict(), id="from_dict"),
            pytest.param((), dict(foo="bar", baz=1), id="from_kwargs"),
            pytest.param((dict(foo="bar"),), dict(baz=1), id="mixed"),
        ],
    )
    def test_instantiation(self, args, kwargs):
        FrozenMapping(*args, **kwargs)

    def test_unhashable_items(self):
        with pytest.raises(TypeError, match="unhashable type"):
            FrozenMapping(foo=[])

    def test_getitem(self):
        options = dict(foo="bar", baz=1)
        config = FrozenMapping(options)

        for key, value in options.items():
            assert config[key] == value

    def test_getitem_unknown(self):
        with pytest.raises(KeyError):
            FrozenMapping()["unknown"]

    def test_iter(self):
        options = dict(foo="bar", baz=1)
        assert set(iter(FrozenMapping(options))) == set(options.keys())

    def test_len(self):
        options = dict(foo="bar", baz=1)
        assert len(FrozenMapping(options)) == len(options)

    def test_immutable_setitem(self):
        frozen_mapping = FrozenMapping()

        with pytest.raises(RuntimeError, match="immutable"):
            frozen_mapping["foo"] = "bar"

    def test_immutable_delitem(
        self,
    ):
        frozen_mapping = FrozenMapping(foo="bar")

        with pytest.raises(RuntimeError, match="immutable"):
            del frozen_mapping["foo"]

    def test_eq(self):
        options = dict(foo="bar", baz=1)
        assert FrozenMapping(options) == FrozenMapping(options)

    def test_ne(self):
        options1 = dict(foo="bar", baz=1)
        options2 = options1.copy()
        options2["baz"] += 1

        assert FrozenMapping(options1) != FrozenMapping(options2)

    def test_repr(self):
        options = dict(foo="bar", baz=1)
        output = repr(FrozenMapping(options))

        assert isinstance(output, str)
        for key, value in options.items():
            assert str(key) in output and str(value) in output


class TestFrozenBunch:
    def test_getattr(self):
        options = dict(foo="bar", baz=1)
        config = FrozenBunch(options)

        for key, value in options.items():
            assert getattr(config, key) == value

    def test_getattr_unknown(self):
        with pytest.raises(AttributeError, match="no attribute 'unknown'"):
            datasets.utils.DatasetConfig().unknown

    def test_immutable_setattr(self):
        frozen_bunch = FrozenBunch()

        with pytest.raises(RuntimeError, match="immutable"):
            frozen_bunch.foo = "bar"

    def test_immutable_delattr(
        self,
    ):
        frozen_bunch = FrozenBunch(foo="bar")

        with pytest.raises(RuntimeError, match="immutable"):
            del frozen_bunch.foo

    def test_repr(self):
        options = dict(foo="bar", baz=1)
        output = repr(FrozenBunch(options))

        assert isinstance(output, str)
        assert output.startswith("FrozenBunch")
        for key, value in options.items():
            assert f"{key}={value}" in output


class TestDatasetInfo:
    @pytest.fixture
    def info(self):
        return make_minimal_dataset_info(valid_options=dict(split=("train", "test"), foo=("bar", "baz")))

    def test_default_config(self, info):
        valid_options = info._valid_options
        default_config = datasets.utils.DatasetConfig({key: values[0] for key, values in valid_options.items()})

        assert info.default_config == default_config

    @pytest.mark.parametrize(
        ("valid_options", "options", "expected_error_msg"),
        [
            (dict(), dict(any_option=None), "does not take any options"),
            (dict(split="train"), dict(unknown_option=None), "Unknown option 'unknown_option'"),
            (dict(split="train"), dict(split="invalid_argument"), "Invalid argument 'invalid_argument'"),
        ],
    )
    def test_make_config_invalid_inputs(self, info, valid_options, options, expected_error_msg):
        info = make_minimal_dataset_info(valid_options=valid_options)

        with pytest.raises(ValueError, match=expected_error_msg):
            info.make_config(**options)

    def test_check_dependencies(self):
        dependency = "fake_dependency"
        info = make_minimal_dataset_info(dependencies=(dependency,))
        with pytest.raises(ModuleNotFoundError, match=dependency):
            info.check_dependencies()

    def test_repr(self, info):
        output = repr(info)

        assert isinstance(output, str)
        assert "DatasetInfo" in output
        for key, value in info._valid_options.items():
            assert f"{key}={str(value)[1:-1]}" in output

    @pytest.mark.parametrize("optional_info", ("citation", "homepage", "license"))
    def test_repr_optional_info(self, optional_info):
        sentinel = "sentinel"
        info = make_minimal_dataset_info(**{optional_info: sentinel})

        assert f"{optional_info}={sentinel}" in repr(info)


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
