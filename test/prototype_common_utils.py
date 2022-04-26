"""This module is separated from common_utils.py to prevent the former to be dependent on torchvision.prototype"""

import PIL.Image
import torch
from torch.testing._comparison import (
    NonePair,
    BooleanPair,
    NumberPair,
    assert_equal as _assert_equal,
    TensorLikePair,
    UnsupportedInputs,
)
from torchvision.prototype import features
from torchvision.transforms.functional_tensor import _max_value as get_max_value

__all__ = ["assert_close"]


class PILImagePair(TensorLikePair):
    def __init__(
        self,
        actual,
        expected,
        *,
        agg_method=None,
        allowed_percentage_diff=None,
        **other_parameters,
    ):
        if not any(isinstance(input, PIL.Image.Image) for input in (actual, expected)):
            raise UnsupportedInputs()

        actual, expected = [
            features.Image(input) if isinstance(input, PIL.Image.Image) else input for input in (actual, expected)
        ]

        super().__init__(actual, expected, **other_parameters)
        self.agg_method = getattr(torch, agg_method) if isinstance(agg_method, str) else agg_method
        self.allowed_percentage_diff = allowed_percentage_diff

        # TODO: comment
        self.check_dtype = False
        self.check_device = False

    def _equalize_attributes(self, actual, expected):
        actual, expected = [input.to(torch.float64).div_(get_max_value(input.dtype)) for input in [actual, expected]]
        return super()._equalize_attributes(actual, expected)

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        if all(isinstance(input, features.Image) for input in (actual, expected)):
            if actual.color_space != expected.color_space:
                self._make_error_meta(AssertionError, "color space mismatch")

        actual, expected = self._equalize_attributes(actual, expected)
        abs_diff = torch.abs(actual - expected)

        if self.allowed_percentage_diff is not None:
            percentage_diff = (abs_diff != 0).to(torch.float).mean()
            if percentage_diff > self.allowed_percentage_diff:
                self._make_error_meta(AssertionError, "percentage mismatch")

        if self.agg_method is None:
            super()._compare_values(actual, expected)
        else:
            err = self.agg_method(abs_diff)
            if err > self.atol:
                self._make_error_meta(AssertionError, "aggregated mismatch")


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
    **kwargs,
):
    """Superset of :func:`torch.testing.assert_close` with support for PIL vs. tensor image comparison"""
    __tracebackhide__ = True

    _assert_equal(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            PILImagePair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
        **kwargs,
    )
