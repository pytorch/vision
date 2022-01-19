import pytest
from torchvision.prototype.utils._internal import sequence_to_str


@pytest.mark.parametrize(
    ("seq", "separate_last", "expected"),
    [
        pytest.param([], "", "", id="empty"),
        pytest.param(["foo"], "", "'foo'", id="single"),
        pytest.param(["foo", "bar"], "", "'foo', 'bar'", id="double"),
        pytest.param(["foo", "bar"], "and ", "'foo' and 'bar'", id="double-separate_last"),
        pytest.param(["foo", "bar", "baz"], "", "'foo', 'bar', 'baz'", id="multi"),
        pytest.param(["foo", "bar", "baz"], "and ", "'foo', 'bar', and 'baz'", id="multi-separate_last"),
    ],
)
def test_sequence_to_str(seq, separate_last, expected):
    assert sequence_to_str(seq, separate_last=separate_last) == expected
