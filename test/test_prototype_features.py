import functools
import itertools

import pytest
import torch
from torch.testing import make_tensor as _make_tensor, assert_close
from torchvision.prototype import features
from torchvision.prototype.utils._internal import sequence_to_str


make_tensor = functools.partial(_make_tensor, device="cpu", dtype=torch.float32)


def make_image(**kwargs):
    data = make_tensor((3, *torch.randint(16, 33, (2,)).tolist()))
    return features.Image(data, **kwargs)


def make_bounding_box(*, format="xyxy", image_size=(10, 10)):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]

    height, width = image_size

    if format == features.BoundingBoxFormat.XYXY:
        x1 = torch.randint(0, width // 2, ())
        y1 = torch.randint(0, height // 2, ())
        x2 = torch.randint(int(x1) + 1, width - int(x1), ()) + x1
        y2 = torch.randint(int(y1) + 1, height - int(y1), ()) + y1
        parts = (x1, y1, x2, y2)
    elif format == features.BoundingBoxFormat.XYWH:
        x = torch.randint(0, width // 2, ())
        y = torch.randint(0, height // 2, ())
        w = torch.randint(1, width - int(x), ())
        h = torch.randint(1, height - int(y), ())
        parts = (x, y, w, h)
    elif format == features.BoundingBoxFormat.CXCYWH:
        cx = torch.randint(1, width - 1, ())
        cy = torch.randint(1, height - 1, ())
        w = torch.randint(1, min(int(cx), width - int(cx)), ())
        h = torch.randint(1, min(int(cy), height - int(cy)), ())
        parts = (cx, cy, w, h)
    else:  # format == features.BoundingBoxFormat._SENTINEL:
        parts = make_tensor((4,)).unbind()

    return features.BoundingBox.from_parts(*parts, format=format, image_size=image_size)


MAKE_DATA_MAP = {
    features.Image: make_image,
    features.BoundingBox: make_bounding_box,
}


def make_feature(feature_type, **meta_data):
    maker = MAKE_DATA_MAP.get(feature_type, lambda **meta_data: feature_type(make_tensor(()), **meta_data))
    return maker(**meta_data)


class TestCommon:
    FEATURE_TYPES, NON_DEFAULT_META_DATA = zip(
        *(
            (features.Image, dict(color_space=features.ColorSpace._SENTINEL)),
            (features.Label, dict(category="category")),
            (features.BoundingBox, dict(format=features.BoundingBoxFormat._SENTINEL, image_size=(-1, -1))),
        )
    )
    feature_types = pytest.mark.parametrize(
        "feature_type", FEATURE_TYPES, ids=lambda feature_type: feature_type.__name__
    )
    features = pytest.mark.parametrize(
        "feature",
        [
            pytest.param(make_feature(feature_type, **meta_data), id=feature_type.__name__)
            for feature_type, meta_data in zip(FEATURE_TYPES, NON_DEFAULT_META_DATA)
        ],
    )

    def test_consistency(self):
        builtin_feature_types = {
            name
            for name, feature_type in features.__dict__.items()
            if not name.startswith("_")
            and isinstance(feature_type, type)
            and issubclass(feature_type, features.Feature)
            and feature_type is not features.Feature
        }
        untested_feature_types = builtin_feature_types - {feature_type.__name__ for feature_type in self.FEATURE_TYPES}
        if untested_feature_types:
            raise AssertionError(
                f"The feature(s) {sequence_to_str(sorted(untested_feature_types), separate_last='and ')} "
                f"is/are exposed at `torchvision.prototype.features`, but is/are not tested by `TestCommon`. "
                f"Please add it/them to `TestCommon.FEATURE_TYPES`."
            )

    @features
    def test_meta_data_attribute_access(self, feature):
        for name, value in feature._meta_data.items():
            assert getattr(feature, name) == feature._meta_data[name]

    @feature_types
    def test_torch_function(self, feature_type):
        input = make_feature(feature_type)
        # This can be any Tensor operation besides clone
        output = input + 1

        assert type(output) is torch.Tensor
        assert_close(output, input + 1)

    @feature_types
    def test_clone(self, feature_type):
        input = make_feature(feature_type)
        output = input.clone()

        assert type(output) is feature_type
        assert_close(output, input)
        assert output._meta_data == input._meta_data

    @features
    def test_serialization(self, tmpdir, feature):
        file = tmpdir / "test_serialization.pt"

        torch.save(feature, str(file))
        loaded_feature = torch.load(str(file))

        assert isinstance(loaded_feature, type(feature))
        assert_close(loaded_feature, feature)
        assert loaded_feature._meta_data == feature._meta_data

    @features
    def test_repr(self, feature):
        assert type(feature).__name__ in repr(feature)


class TestBoundingBox:
    @pytest.mark.parametrize(("format", "intermediate_format"), itertools.permutations(("xyxy", "xywh"), 2))
    def test_cycle_consistency(self, format, intermediate_format):
        input = make_bounding_box(format=format)
        output = input.convert(intermediate_format).convert(format)
        assert_close(input, output)


# For now, tensor subclasses with additional meta data do not work with torchscript.
# See https://github.com/pytorch/vision/pull/4721#discussion_r741676037.
@pytest.mark.xfail
class TestJit:
    def test_bounding_box(self):
        def resize(input: features.BoundingBox, size: torch.Tensor) -> features.BoundingBox:
            old_height, old_width = input.image_size
            new_height, new_width = size

            height_scale = new_height / old_height
            width_scale = new_width / old_width

            old_x1, old_y1, old_x2, old_y2 = input.convert("xyxy").to_parts()

            new_x1 = old_x1 * width_scale
            new_y1 = old_y1 * height_scale

            new_x2 = old_x2 * width_scale
            new_y2 = old_y2 * height_scale

            return features.BoundingBox.from_parts(
                new_x1, new_y1, new_x2, new_y2, like=input, format="xyxy", image_size=tuple(size.tolist())
            )

        def horizontal_flip(input: features.BoundingBox) -> features.BoundingBox:
            x, y, w, h = input.convert("xywh").to_parts()
            x = input.image_size[1] - (x + w)
            return features.BoundingBox.from_parts(x, y, w, h, like=input, format="xywh")

        def compose(input: features.BoundingBox, size: torch.Tensor) -> features.BoundingBox:
            return horizontal_flip(resize(input, size)).convert("xyxy")

        image_size = (8, 6)
        input = features.BoundingBox([2, 4, 2, 4], format="cxcywh", image_size=image_size)
        size = torch.tensor((4, 12))
        expected = features.BoundingBox([6, 1, 10, 3], format="xyxy", image_size=image_size)

        actual_eager = compose(input, size)
        assert_close(actual_eager, expected)

        sample_inputs = (features.BoundingBox(torch.zeros((4,)), image_size=(10, 10)), torch.tensor((20, 5)))
        actual_jit = torch.jit.trace(compose, sample_inputs, check_trace=False)(input, size)
        assert_close(actual_jit, expected)
