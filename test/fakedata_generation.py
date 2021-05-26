"""This module is deprecated and only kept for BC with FB internal code."""

import contextlib

from test_datasets import CIFAR10TestCase, CIFAR100
from common_utils import get_tmp_dir


@contextlib.contextmanager
def dataset_root_from_test_case(test_case_cls, config=None):
    test_case_cls.setUpClass()

    if config is None:
        config = test_case_cls._KWARG_DEFAULTS
        if test_case_cls.DEFAULT_CONFIG:
            config.update(test_case_cls.DEFAULT_CONFIG)

    with get_tmp_dir() as root:
        test_case = test_case_cls()
        test_case.inject_fake_data(root, config)
        yield root


@contextlib.contextmanager
def cifar_root(version):
    with dataset_root_from_test_case(CIFAR10TestCase if version == "CIFAR10" else CIFAR100) as root:
        yield root
