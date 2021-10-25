import functools

import pytest
import torch
from torch import jit
from torch.testing import make_tensor as _make_tensor, assert_close
from torchvision.prototype import features

make_tensor = functools.partial(_make_tensor, device="cpu", dtype=torch.float32)


class TestJIT:
    FEATURE_TYPES = {
        feature_type
        for name, feature_type in features.__dict__.items()
        if not name.startswith("_")
        and isinstance(feature_type, type)
        and issubclass(feature_type, features.Feature)
        and feature_type is not features.Feature
    }
    feature_types = pytest.mark.parametrize(
        "feature_type", FEATURE_TYPES, ids=lambda feature_type: feature_type.__name__
    )

    @feature_types
    def test_identity(self, feature_type):
        def identity(input):
            return input

        identity.__annotations__ = {"input": feature_type, "return": feature_type}

        scripted_fn = jit.script(identity)
        input = feature_type(make_tensor(()))
        output = scripted_fn(input)

        assert output is input

    @feature_types
    def test_torch_function(self, feature_type):
        def any_operation_except_clone(input) -> torch.Tensor:
            return input + 0

        any_operation_except_clone.__annotations__["input"] = feature_type

        scripted_fn = jit.script(any_operation_except_clone)
        input = feature_type(make_tensor(()))
        output = scripted_fn(input)

        assert type(output) is torch.Tensor

    @feature_types
    def test_clone(self, feature_type):
        def clone(input):
            return input.clone()

        clone.__annotations__ = {"input": feature_type, "return": feature_type}

        scripted_fn = jit.script(clone)
        input = feature_type(make_tensor(()))
        output = scripted_fn(input)

        assert type(output) is feature_type
        assert_close(output, input)
