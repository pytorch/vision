import collections.abc
import dataclasses
from typing import Optional, Sequence

import pytest
import torch

from common_utils import combinations_grid, DEFAULT_EXTRA_DIMS, from_loader, from_loaders, TensorLoader
from torch.nn.functional import one_hot

from torchvision.prototype import datapoints


@dataclasses.dataclass
class LabelLoader(TensorLoader):
    categories: Optional[Sequence[str]]


def _parse_categories(categories):
    if categories is None:
        num_categories = int(torch.randint(1, 11, ()))
    elif isinstance(categories, int):
        num_categories = categories
        categories = [f"category{idx}" for idx in range(num_categories)]
    elif isinstance(categories, collections.abc.Sequence) and all(isinstance(category, str) for category in categories):
        categories = list(categories)
        num_categories = len(categories)
    else:
        raise pytest.UsageError(
            f"`categories` can either be `None` (default), an integer, or a sequence of strings, "
            f"but got '{categories}' instead."
        )
    return categories, num_categories


def make_label_loader(*, extra_dims=(), categories=None, dtype=torch.int64):
    categories, num_categories = _parse_categories(categories)

    def fn(shape, dtype, device):
        # The idiom `make_tensor(..., dtype=torch.int64).to(dtype)` is intentional to only get integer values,
        # regardless of the requested dtype, e.g. 0 or 0.0 rather than 0 or 0.123
        data = torch.testing.make_tensor(shape, low=0, high=num_categories, dtype=torch.int64, device=device).to(dtype)
        return datapoints.Label(data, categories=categories)

    return LabelLoader(fn, shape=extra_dims, dtype=dtype, categories=categories)


make_label = from_loader(make_label_loader)


@dataclasses.dataclass
class OneHotLabelLoader(TensorLoader):
    categories: Optional[Sequence[str]]


def make_one_hot_label_loader(*, categories=None, extra_dims=(), dtype=torch.int64):
    categories, num_categories = _parse_categories(categories)

    def fn(shape, dtype, device):
        if num_categories == 0:
            data = torch.empty(shape, dtype=dtype, device=device)
        else:
            # The idiom `make_label_loader(..., dtype=torch.int64); ...; one_hot(...).to(dtype)` is intentional
            # since `one_hot` only supports int64
            label = make_label_loader(extra_dims=extra_dims, categories=num_categories, dtype=torch.int64).load(device)
            data = one_hot(label, num_classes=num_categories).to(dtype)
        return datapoints.OneHotLabel(data, categories=categories)

    return OneHotLabelLoader(fn, shape=(*extra_dims, num_categories), dtype=dtype, categories=categories)


def make_one_hot_label_loaders(
    *,
    categories=(1, 0, None),
    extra_dims=DEFAULT_EXTRA_DIMS,
    dtypes=(torch.int64, torch.float32),
):
    for params in combinations_grid(categories=categories, extra_dims=extra_dims, dtype=dtypes):
        yield make_one_hot_label_loader(**params)


make_one_hot_labels = from_loaders(make_one_hot_label_loaders)
