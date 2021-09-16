from torchvision.prototype.datasets import utils

import pytest


class TestDatasetConfig:
    @staticmethod
    @pytest.fixture
    def options():
        return dict(foo="bar", baz=1)

    def test_creation_from_dict(self, options):
        utils.DatasetConfig(options)

    def test_creation_from_kwargs(self, options):
        utils.DatasetConfig(**options)

    def test_creation_mixed(self, options):
        utils.DatasetConfig(options, baz=options.pop("baz"))

    def test_creation_unhashable(self, options):
        with pytest.raises(TypeError):
            utils.DatasetConfig(options=options)

    def test_getitem(self, options):
        config = utils.DatasetConfig(options)

        for key, value in options.items():
            assert config[key] == value

    def test_getitem_unknown(self):
        with pytest.raises(KeyError):
            utils.DatasetConfig()["unknown"]

    def test_iter(self, options):
        assert set(iter(utils.DatasetConfig(options))) == set(options.keys())

    def test_len(self, options):
        assert len(utils.DatasetConfig(options)) == len(options)

    def test_getattr(self, options):
        config = utils.DatasetConfig(options)

        for key, value in options.items():
            assert getattr(config, key) == value

    def test_getattr_unknown(self):
        with pytest.raises(AttributeError):
            utils.DatasetConfig().unknown

    def test_setitem(self):
        config = utils.DatasetConfig()

        with pytest.raises(RuntimeError):
            config["foo"] = "bar"

    def test_setattr(self):
        config = utils.DatasetConfig()

        with pytest.raises(RuntimeError):
            config.foo = "bar"

    def test_delitem(self, options):
        config = utils.DatasetConfig(options)

        with pytest.raises(RuntimeError):
            del config["foo"]

    def test_delattr(self, options):
        config = utils.DatasetConfig(options)

        with pytest.raises(RuntimeError):
            del config.foo

    def test_eq(self, options):
        config1 = utils.DatasetConfig(options)
        config2 = utils.DatasetConfig(options)

        assert config1 == config2

    def test_repr(self, options):
        output = repr(utils.DatasetConfig(options))

        assert isinstance(output, str)
        assert "DatasetConfig" in output
        for key, value in options.items():
            assert f"{key}={value}" in output

    def test_contains(self, options):
        config = utils.DatasetConfig(options)

        for key in options.keys():
            assert key in config

    def test_keys(self, options):
        assert set(utils.DatasetConfig(options).keys()) == set(options.keys())

    def test_values(self, options):
        assert set(utils.DatasetConfig(options).values()) == set(options.values())

    def test_get(self, options):
        config = utils.DatasetConfig(options)

        for key, value in options.items():
            assert config.get(key) == value

    def test_get_default(self, options):
        sentinel = object()

        assert utils.DatasetConfig().get("unknown", sentinel) is sentinel

    def test_ne(self, options):
        config1 = utils.DatasetConfig(options)
        options["baz"] = 2
        config2 = utils.DatasetConfig(options)

        assert config1 != config2


class TestDatasetInfo:
    @staticmethod
    @pytest.fixture
    def options():
        return dict(split=("train", "test"), foo=("bar", "baz"))

    def test_default_config(self, options):
        default_config = utils.DatasetConfig({key: values[0] for key, values in options.items()})

        assert utils.DatasetInfo("name", options=options).default_config == default_config

    def test_repr(self, options):
        output = repr(utils.DatasetInfo("name", options=options))

        assert isinstance(output, str)
        assert "DatasetInfo" in output
        for key, value in options.items():
            assert f"{key}={value}" in output

    @pytest.mark.parametrize("optional_info", ("citation", "homepage", "license"))
    def test_repr_optional_info(self, optional_info):
        sentinel = "sentinel"
        info = utils.DatasetInfo("name", **{optional_info: sentinel})

        assert f"{optional_info}={sentinel}" in repr(info)
