import random

import numpy as np
import pytest
import torch

from common_utils import (
    CUDA_NOT_AVAILABLE_MSG,
    IN_FBCODE,
    IN_OSS_CI,
    IN_RE_WORKER,
    MPS_NOT_AVAILABLE_MSG,
    OSS_CI_GPU_NO_CUDA_MSG,
)


def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line("markers", "needs_cuda: mark for tests that rely on a CUDA device")
    config.addinivalue_line("markers", "needs_mps: mark for tests that rely on a MPS device")
    config.addinivalue_line("markers", "dont_collect: mark for tests that should not be collected")
    config.addinivalue_line("markers", "opcheck_only_one: only opcheck one parametrization")


def pytest_collection_modifyitems(items):
    # This hook is called by pytest after it has collected the tests (google its name to check out its doc!)
    # We can ignore some tests as we see fit here, or add marks, such as a skip mark.
    #
    # Typically, here, we try to optimize CI time. In particular, the GPU CI instances don't need to run the
    # tests that don't need CUDA, because those tests are extensively tested in the CPU CI instances already.
    # This is true for both OSS CI and the fbcode internal CI.
    # In the fbcode CI, we have an additional constraint: we try to avoid skipping tests. So instead of relying on
    # pytest.mark.skip, in fbcode we literally just remove those tests from the `items` list, and it's as if
    # these tests never existed.

    out_items = []
    for item in items:
        # The needs_cuda mark will exist if the test was explicitly decorated with
        # the @needs_cuda decorator. It will also exist if it was parametrized with a
        # parameter that has the mark: for example if a test is parametrized with
        # @pytest.mark.parametrize('device', cpu_and_cuda())
        # the "instances" of the tests where device == 'cuda' will have the 'needs_cuda' mark,
        # and the ones with device == 'cpu' won't have the mark.
        needs_cuda = item.get_closest_marker("needs_cuda") is not None
        needs_mps = item.get_closest_marker("needs_mps") is not None

        if needs_cuda and not torch.cuda.is_available():
            # In general, we skip cuda tests on machines without a GPU
            # There are special cases though, see below
            item.add_marker(pytest.mark.skip(reason=CUDA_NOT_AVAILABLE_MSG))

        if needs_mps and not torch.backends.mps.is_available():
            item.add_marker(pytest.mark.skip(reason=MPS_NOT_AVAILABLE_MSG))

        if IN_FBCODE:
            # fbcode doesn't like skipping tests, so instead we  just don't collect the test
            # so that they don't even "exist", hence the continue statements.
            if not needs_cuda and IN_RE_WORKER:
                # The RE workers are the machines with GPU, we don't want them to run CPU-only tests.
                continue
            if needs_cuda and not torch.cuda.is_available():
                # On the test machines without a GPU, we want to ignore the tests that need cuda.
                # TODO: something more robust would be to do that only in a sandcastle instance,
                # so that we can still see the test being skipped when testing locally from a devvm
                continue
            if needs_mps and not torch.backends.mps.is_available():
                # Same as above, but for MPS
                continue
        elif IN_OSS_CI:
            # Here we're not in fbcode, so we can safely collect and skip tests.
            if not needs_cuda and torch.cuda.is_available():
                # Similar to what happens in RE workers: we don't need the OSS CI GPU machines
                # to run the CPU-only tests.
                item.add_marker(pytest.mark.skip(reason=OSS_CI_GPU_NO_CUDA_MSG))

        if item.get_closest_marker("dont_collect") is not None:
            # currently, this is only used for some tests we're sure we don't want to run on fbcode
            continue

        out_items.append(item)

    items[:] = out_items


def pytest_sessionfinish(session, exitstatus):
    # This hook is called after all tests have run, and just before returning an exit status.
    # We here change exit code 5 into 0.
    #
    # 5 is issued when no tests were actually run, e.g. if you use `pytest -k some_regex_that_is_never_matched`.
    #
    # Having no test being run for a given test rule is a common scenario in fbcode, and typically happens on
    # the GPU test machines which don't run the CPU-only tests (see pytest_collection_modifyitems above). For
    # example `test_transforms.py` doesn't contain any CUDA test at the time of
    # writing, so on a GPU test machine, testpilot would invoke pytest on this file and no test would be run.
    # This would result in pytest returning 5, causing testpilot to raise an error.
    # To avoid this, we transform this 5 into a 0 to make testpilot happy.
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(autouse=True)
def prevent_leaking_rng():
    # Prevent each test from leaking the rng to all other test when they call
    # torch.manual_seed() or random.seed() or np.random.seed().
    # Note: the numpy rngs should never leak anyway, as we never use
    # np.random.seed() and instead rely on np.random.RandomState instances (see
    # issue #4247). We still do it for extra precaution.

    torch_rng_state = torch.get_rng_state()
    builtin_rng_state = random.getstate()
    nunmpy_rng_state = np.random.get_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(torch_rng_state)
    random.setstate(builtin_rng_state)
    np.random.set_state(nunmpy_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
