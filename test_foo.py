import pytest


def test_success():
    assert True


def test_failure():
    assert False


@pytest.mark.skip
def test_skip():
    pass


@pytest.mark.xfail
def test_xfail():
    assert False


@pytest.mark.xfail
def test_pass():
    assert True
