def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line(
        "markers", "dont_collect: marks a test that should not be collected (avoids skipping it)"
    )


def pytest_collection_modifyitems(items):
    # This hook is called by pytest after it has collected the tests (google its name!)
    # We can ignore some tests as we see fit here. In particular we ignore the tests that
    # we have marked with the custom 'dont_collect' mark. This avoids skipping the tests,
    # since the internal fb infra doesn't like skipping tests.
    to_keep = [item for item in items if item.get_closest_marker('dont_collect') is None]
    items[:] = to_keep
