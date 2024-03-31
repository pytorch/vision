import pytest
from torchvision._utils import sequence_to_str


@pytest.mark.parametrize(
    ("seq", "separate_last", "expected"),
    [
        ([], "", ""),
        (["foo"], "", "'foo'"),
        (["foo", "bar"], "", "'foo', 'bar'"),
        (["foo", "bar"], "and ", "'foo' and 'bar'"),
        (["foo", "bar", "baz"], "", "'foo', 'bar', 'baz'"),
        (["foo", "bar", "baz"], "and ", "'foo', 'bar', and 'baz'"),
    ],
)
def test_sequence_to_str(seq, separate_last, expected):
    assert sequence_to_str(seq, separate_last=separate_last) == expected
