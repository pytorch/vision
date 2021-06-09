from common_utils import IN_CIRCLE_CI, CIRCLECI_GPU_NO_CUDA_MSG
import torch
import pytest


def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line(
        "markers", "dont_collect: marks a test that should not be collected (avoids skipping it)"
    )
    config.addinivalue_line(
        "markers", "needs_cuda: mark for tests that rely on a CUDA device"
    )


def pytest_collection_modifyitems(items):
    # This hook is called by pytest after it has collected the tests (google its name!)
    # We can ignore some tests as we see fit here. In particular we ignore the tests that
    # we have marked with the custom 'dont_collect' mark. This avoids skipping the tests,
    # since the internal fb infra doesn't like skipping tests.

    # Also, we look at the tests that are marked with a needs_cuda mark, and those that aren't.
    # If a test doesn't need cuda but we're in a CircleCI GPU machine, we don't run the test,
    # as it's been run by the cpu workflows already.

    out_items = []
    for item in items:
        collect = item.get_closest_marker('dont_collect') is None
        if not collect:
            # TODO: We should be able to get rid of the dont_collect mark altogether
            continue
        needs_cuda = item.get_closest_marker('needs_cuda') is None
        if IN_CIRCLE_CI:
            if not needs_cuda and torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason=CIRCLECI_GPU_NO_CUDA_MSG))
        # TODO: do the same for fbcode
        out_items.append(item)

    return out_items
