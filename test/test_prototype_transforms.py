import pytest
from torchvision.prototype import transforms, features
from torchvision.prototype.utils._internal import sequence_to_str


FEATURE_TYPES = {
    feature_type
    for name, feature_type in features.__dict__.items()
    if not name.startswith("_")
    and isinstance(feature_type, type)
    and issubclass(feature_type, features.Feature)
    and feature_type is not features.Feature
}

TRANSFORM_TYPES = tuple(
    transform_type
    for name, transform_type in transforms.__dict__.items()
    if not name.startswith("_")
    and isinstance(transform_type, type)
    and issubclass(transform_type, transforms.Transform)
    and transform_type is not transforms.Transform
)


def test_feature_type_support():
    missing_feature_types = FEATURE_TYPES - set(transforms.Transform._BUILTIN_FEATURE_TYPES)
    if missing_feature_types:
        names = sorted([feature_type.__name__ for feature_type in missing_feature_types])
        raise AssertionError(
            f"The feature(s) {sequence_to_str(names, separate_last='and ')} is/are exposed at "
            f"`torchvision.prototype.features`, but are missing in Transform._BUILTIN_FEATURE_TYPES. "
            f"Please add it/them to the collection."
        )


@pytest.mark.parametrize(
    "transform_type",
    [transform_type for transform_type in TRANSFORM_TYPES if transform_type is not transforms.Identity],
    ids=lambda transform_type: transform_type.__name__,
)
def test_feature_no_op_coverage(transform_type):
    unsupported_features = (
        FEATURE_TYPES - transform_type.supported_feature_types() - set(transform_type.NO_OP_FEATURE_TYPES)
    )
    if unsupported_features:
        names = sorted([feature_type.__name__ for feature_type in unsupported_features])
        raise AssertionError(
            f"The feature(s) {sequence_to_str(names, separate_last='and ')} are neither supported nor declared as "
            f"no-op for transform `{transform_type.__name__}`. Please either implement a feature transform for them, "
            f"or add them to the the `{transform_type.__name__}.NO_OP_FEATURE_TYPES` collection."
        )


def test_non_feature_no_op():
    class TestTransform(transforms.Transform):
        @staticmethod
        def image(input):
            return input

    no_op_sample = dict(int=0, float=0.0, bool=False, str="str")
    assert TestTransform()(no_op_sample) == no_op_sample
