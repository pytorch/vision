import functools
import itertools

import PIL.Image
import pytest

import torch
import torch.testing
from torch.nn.functional import one_hot
from torch.testing._comparison import assert_equal as _assert_equal, TensorLikePair
from torchvision.prototype import features
from torchvision.prototype.transforms.functional import to_image_tensor
from torchvision.transforms.functional_tensor import _max_value as get_max_value


class ImagePair(TensorLikePair):
    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        return super()._process_inputs(
            *[to_image_tensor(input) if isinstance(input, PIL.Image.Image) else input for input in [actual, expected]],
            id=id,
            allow_subclasses=allow_subclasses,
        )


assert_equal = functools.partial(_assert_equal, pair_types=[ImagePair], rtol=0, atol=0)


class ArgsKwargs:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs

    def __str__(self):
        def short_repr(obj, max=20):
            repr_ = repr(obj)
            if len(repr_) <= max:
                return repr_

            return f"{repr_[:max//2]}...{repr_[-(max//2-3):]}"

        return ", ".join(
            itertools.chain(
                [short_repr(arg) for arg in self.args],
                [f"{param}={short_repr(kwarg)}" for param, kwarg in self.kwargs.items()],
            )
        )


make_tensor = functools.partial(torch.testing.make_tensor, device="cpu")


def make_image(size=None, *, color_space, extra_dims=(), dtype=torch.float32, constant_alpha=True):
    size = size or torch.randint(16, 33, (2,)).tolist()

    try:
        num_channels = {
            features.ColorSpace.GRAY: 1,
            features.ColorSpace.GRAY_ALPHA: 2,
            features.ColorSpace.RGB: 3,
            features.ColorSpace.RGB_ALPHA: 4,
        }[color_space]
    except KeyError as error:
        raise pytest.UsageError() from error

    shape = (*extra_dims, num_channels, *size)
    max_value = get_max_value(dtype)
    data = make_tensor(shape, low=0, high=max_value, dtype=dtype)
    if color_space in {features.ColorSpace.GRAY_ALPHA, features.ColorSpace.RGB_ALPHA} and constant_alpha:
        data[..., -1, :, :] = max_value
    return features.Image(data, color_space=color_space)


make_grayscale_image = functools.partial(make_image, color_space=features.ColorSpace.GRAY)
make_rgb_image = functools.partial(make_image, color_space=features.ColorSpace.RGB)


def make_images(
    sizes=((16, 16), (7, 33), (31, 9)),
    color_spaces=(
        features.ColorSpace.GRAY,
        features.ColorSpace.GRAY_ALPHA,
        features.ColorSpace.RGB,
        features.ColorSpace.RGB_ALPHA,
    ),
    dtypes=(torch.float32, torch.uint8),
    extra_dims=((), (0,), (4,), (2, 3), (5, 0), (0, 5)),
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image(size, color_space=color_space, dtype=dtype)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image(size=sizes[0], color_space=color_space, extra_dims=extra_dims_, dtype=dtype)


def randint_with_tensor_bounds(arg1, arg2=None, **kwargs):
    low, high = torch.broadcast_tensors(
        *[torch.as_tensor(arg) for arg in ((0, arg1) if arg2 is None else (arg1, arg2))]
    )
    return torch.stack(
        [
            torch.randint(low_scalar, high_scalar, (), **kwargs)
            for low_scalar, high_scalar in zip(low.flatten().tolist(), high.flatten().tolist())
        ]
    ).reshape(low.shape)


def make_bounding_box(*, format, image_size=(32, 32), extra_dims=(), dtype=torch.int64):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]

    if any(dim == 0 for dim in extra_dims):
        return features.BoundingBox(torch.empty(*extra_dims, 4), format=format, image_size=image_size)

    height, width = image_size

    if format == features.BoundingBoxFormat.XYXY:
        x1 = torch.randint(0, width // 2, extra_dims)
        y1 = torch.randint(0, height // 2, extra_dims)
        x2 = randint_with_tensor_bounds(x1 + 1, width - x1) + x1
        y2 = randint_with_tensor_bounds(y1 + 1, height - y1) + y1
        parts = (x1, y1, x2, y2)
    elif format == features.BoundingBoxFormat.XYWH:
        x = torch.randint(0, width // 2, extra_dims)
        y = torch.randint(0, height // 2, extra_dims)
        w = randint_with_tensor_bounds(1, width - x)
        h = randint_with_tensor_bounds(1, height - y)
        parts = (x, y, w, h)
    elif format == features.BoundingBoxFormat.CXCYWH:
        cx = torch.randint(1, width - 1, ())
        cy = torch.randint(1, height - 1, ())
        w = randint_with_tensor_bounds(1, torch.minimum(cx, width - cx) + 1)
        h = randint_with_tensor_bounds(1, torch.minimum(cy, height - cy) + 1)
        parts = (cx, cy, w, h)
    else:
        raise pytest.UsageError()

    return features.BoundingBox(torch.stack(parts, dim=-1).to(dtype), format=format, image_size=image_size)


make_xyxy_bounding_box = functools.partial(make_bounding_box, format=features.BoundingBoxFormat.XYXY)


def make_bounding_boxes(
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_sizes=((32, 32),),
    dtypes=(torch.int64, torch.float32),
    extra_dims=((0,), (), (4,), (2, 3), (5, 0), (0, 5)),
):
    for format, image_size, dtype in itertools.product(formats, image_sizes, dtypes):
        yield make_bounding_box(format=format, image_size=image_size, dtype=dtype)

    for format, extra_dims_ in itertools.product(formats, extra_dims):
        yield make_bounding_box(format=format, extra_dims=extra_dims_)


def make_label(size=(), *, categories=("category0", "category1")):
    return features.Label(torch.randint(0, len(categories) if categories else 10, size), categories=categories)


def make_one_hot_label(*args, **kwargs):
    label = make_label(*args, **kwargs)
    return features.OneHotLabel(one_hot(label, num_classes=len(label.categories)), categories=label.categories)


def make_one_hot_labels(
    *,
    num_categories=(1, 2, 10),
    extra_dims=((), (0,), (4,), (2, 3), (5, 0), (0, 5)),
):
    for num_categories_ in num_categories:
        yield make_one_hot_label(categories=[f"category{idx}" for idx in range(num_categories_)])

    for extra_dims_ in extra_dims:
        yield make_one_hot_label(extra_dims_)


def make_segmentation_mask(size=None, *, num_objects=None, extra_dims=(), dtype=torch.uint8):
    size = size if size is not None else torch.randint(16, 33, (2,)).tolist()
    num_objects = num_objects if num_objects is not None else int(torch.randint(1, 11, ()))
    shape = (*extra_dims, num_objects, *size)
    data = make_tensor(shape, low=0, high=2, dtype=dtype)
    return features.SegmentationMask(data)


def make_segmentation_masks(
    sizes=((16, 16), (7, 33), (31, 9)),
    dtypes=(torch.uint8,),
    extra_dims=((), (0,), (4,), (2, 3), (5, 0), (0, 5)),
    num_objects=(1, 0, 10),
):
    for size, dtype, extra_dims_ in itertools.product(sizes, dtypes, extra_dims):
        yield make_segmentation_mask(size=size, dtype=dtype, extra_dims=extra_dims_)

    for dtype, extra_dims_, num_objects_ in itertools.product(dtypes, extra_dims, num_objects):
        yield make_segmentation_mask(size=sizes[0], num_objects=num_objects_, dtype=dtype, extra_dims=extra_dims_)
