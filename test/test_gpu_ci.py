import os

from common_utils import needs_cuda, cpu_and_gpu
import pytest


@pytest.mark.parametrize("device", cpu_and_gpu())
def test_cpu_and_gpu(request, device):
    print(os.getenv("GITHUB_ACTIONS"))
    for mark in request.node.own_markers:
        print(mark)
    assert device != "cpu", "This should not be run on a GPU machine"


@needs_cuda
def test_needs_cuda(request):
    for mark in request.node.own_markers:
        print(mark)
