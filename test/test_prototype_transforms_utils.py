import pytest
from torchvision.prototype.transforms._utils import has_all, has_any


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((0, 0.0, ""), (int,), True),
        ((0, 0.0, ""), (float,), True),
        ((0, 0.0, ""), (str,), True),
        ((0, 0.0, ""), (int, float), True),
        ((0, 0.0, ""), (int, str), True),
        ((0, 0.0, ""), (float, str), True),
        (("",), (int, float), False),
        ((0.0,), (int, str), False),
        ((0,), (float, str), False),
        ((0, 0.0, ""), (int, float, str), True),
        ((), (int, float, str), False),
        ((0, 0.0, ""), (lambda obj: isinstance(obj, int),), True),
        ((0, 0.0, ""), (lambda _: False,), False),
        ((0, 0.0, ""), (lambda _: True,), True),
    ],
)
def test_has_any(sample, types, expected):
    assert has_any(sample, *types) is expected


@pytest.mark.parametrize(
    ("sample", "types", "expected"),
    [
        ((0, 0.0, ""), (int,), True),
        ((0, 0.0, ""), (float,), True),
        ((0, 0.0, ""), (str,), True),
        ((0, 0.0, ""), (int, float), True),
        ((0, 0.0, ""), (int, str), True),
        ((0, 0.0, ""), (float, str), True),
        ((0, 0.0, ""), (int, float, str), True),
        ((0.0, ""), (int, float), False),
        ((0.0, ""), (int, str), False),
        ((0, ""), (float, str), False),
        ((0, 0.0, ""), (int, float, str), True),
        ((0.0, ""), (int, float, str), False),
        ((0, ""), (int, float, str), False),
        ((0, 0.0), (int, float, str), False),
        ((0, 0.0, ""), (lambda obj: isinstance(obj, (int, float, str)),), True),
        ((0, 0.0, ""), (lambda _: False,), False),
        ((0, 0.0, ""), (lambda _: True,), True),
    ],
)
def test_has_all(sample, types, expected):
    assert has_all(sample, *types) is expected
