import os

from common_utils import needs_cuda, cpu_and_gpu
import pytest


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_cpu_and_gpu(device):
    print(os.getenv("GITHUB_ACTIONS"))
    assert device != "cpu", "This should not be run on a GPU machine"


@needs_cuda
def test_needs_cuda():
    assert True
