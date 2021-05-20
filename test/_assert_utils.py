"""This is a temporary module and should be removed as soon as torch.testing.assert_equal is supported."""
# TODO: remove this as soon torch.testing.assert_equal is supported

import functools

import torch.testing

__all__ = ["assert_equal"]


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
