import functools

import pytest
import torch
from torch.testing import make_tensor as _make_tensor, assert_close
from torchvision.prototype import features
from torchvision.prototype.datasets.utils._internal import FrozenMapping

make_tensor = functools.partial(_make_tensor, device="cpu", dtype=torch.float32)


class TestCommon:
    FEATURE_TYPES, NON_DEFAULT_META_DATA = zip(
        *(
            (features.Image, FrozenMapping(color_space=features.ColorSpace._SENTINEL)),
            (features.Label, FrozenMapping(category="category")),
        )
    )
    feature_types = pytest.mark.parametrize(
        "feature_type", FEATURE_TYPES, ids=lambda feature_type: feature_type.__name__
    )
    feature_types_with_non_default_meta_data = pytest.mark.parametrize(
        ("feature_type", "meta_data"),
        [
            pytest.param(feature_type, FrozenMapping(meta_data), id=feature_type.__name__)
            for feature_type, meta_data in zip(FEATURE_TYPES, NON_DEFAULT_META_DATA)
        ],
    )

    def test_consistency(self):
        builtin_feature_types = {
            feature_type
            for name, feature_type in features.__dict__.items()
            if not name.startswith("_")
            and isinstance(feature_type, type)
            and issubclass(feature_type, features.Feature)
            and feature_type is not features.Feature
        }
        untested_feature_types = builtin_feature_types - set(self.FEATURE_TYPES)
        if untested_feature_types:
            raise AssertionError("FIXME")

    jit_fns = pytest.mark.parametrize(
        "jit_fn",
        [
            pytest.param(lambda fn, example_inputs: fn, id="no_jit"),
            pytest.param(lambda fn, example_inputs: torch.jit.trace(fn, example_inputs), id="torch.jit.trace"),
            pytest.param(lambda fn, example_inputs: torch.jit.script(fn), id="torch.jit.script"),
        ],
    )

    @jit_fns
    @feature_types
    def test_torch_function(self, jit_fn, feature_type):
        def fn(input):
            return input + 1

        input = feature_type(make_tensor(()))

        fn.__annotations__ = {"input": feature_type, "return": torch.Tensor}
        fn = jit_fn(fn, input)

        output = fn(input)

        assert type(output) is torch.Tensor
        assert_close(output, input + 1)

    @jit_fns
    @feature_types
    def test_clone(self, jit_fn, feature_type):
        def fn(input):
            return input.clone()

        fn.__annotations__ = {"input": feature_type, "return": feature_type}
        fn = jit_fn(fn)

        input = feature_type(make_tensor(()))
        output = fn(input)

        assert type(output) is feature_type
        assert_close(output, input)
        assert output._meta_data == input._meta_data

    @feature_types_with_non_default_meta_data
    def test_serialization(self, tmpdir, feature_type, meta_data):
        feature = feature_type(make_tensor(()), **meta_data)
        file = tmpdir / "test_serialization.pt"

        torch.save(feature, str(file))
        loaded_feature = torch.load(str(file))

        assert isinstance(loaded_feature, feature_type)
        assert_close(loaded_feature, feature)
        assert loaded_feature._meta_data == meta_data

    @feature_types
    def test_repr(self, feature_type):
        assert feature_type.__name__ in repr(feature_type(make_tensor(())))
