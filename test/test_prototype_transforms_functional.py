import functools
import itertools

import pytest
import torch.testing
import torchvision.prototype.transforms.functional as F
from torch import jit
from torch.nn.functional import one_hot
from torchvision.prototype import features

make_tensor = functools.partial(torch.testing.make_tensor, device="cpu")


def make_image(size=None, *, color_space, extra_dims=(), dtype=torch.float32):
    size = size or torch.randint(16, 33, (2,)).tolist()

    if isinstance(color_space, str):
        color_space = features.ColorSpace[color_space]
    num_channels = {
        features.ColorSpace.GRAYSCALE: 1,
        features.ColorSpace.RGB: 3,
    }[color_space]

    shape = (*extra_dims, num_channels, *size)
    if dtype.is_floating_point:
        data = torch.rand(shape, dtype=dtype)
    else:
        data = torch.randint(0, torch.iinfo(dtype).max, shape, dtype=dtype)
    return features.Image(data, color_space=color_space)


make_grayscale_image = functools.partial(make_image, color_space=features.ColorSpace.GRAYSCALE)
make_rgb_image = functools.partial(make_image, color_space=features.ColorSpace.RGB)


def make_images(
    sizes=((16, 16), (7, 33), (31, 9)),
    color_spaces=(features.ColorSpace.GRAYSCALE, features.ColorSpace.RGB),
    dtypes=(torch.float32, torch.uint8),
    extra_dims=((4,), (2, 3)),
):
    for size, color_space, dtype in itertools.product(sizes, color_spaces, dtypes):
        yield make_image(size, color_space=color_space, dtype=dtype)

    for color_space, dtype, extra_dims_ in itertools.product(color_spaces, dtypes, extra_dims):
        yield make_image(color_space=color_space, extra_dims=extra_dims_, dtype=dtype)


def randint_with_tensor_bounds(arg1, arg2=None, **kwargs):
    low, high = torch.broadcast_tensors(
        *[torch.as_tensor(arg) for arg in ((0, arg1) if arg2 is None else (arg1, arg2))]
    )
    try:
        return torch.stack(
            [
                torch.randint(low_scalar, high_scalar, (), **kwargs)
                for low_scalar, high_scalar in zip(low.flatten().tolist(), high.flatten().tolist())
            ]
        ).reshape(low.shape)
    except RuntimeError as error:
        raise error


def make_bounding_box(*, format, image_size=(32, 32), extra_dims=(), dtype=torch.int64):
    if isinstance(format, str):
        format = features.BoundingBoxFormat[format]

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
        h = randint_with_tensor_bounds(1, torch.minimum(cy, width - cy) + 1)
        parts = (cx, cy, w, h)
    else:  # format == features.BoundingBoxFormat._SENTINEL:
        raise ValueError()

    return features.BoundingBox(torch.stack(parts, dim=-1).to(dtype), format=format, image_size=image_size)


make_xyxy_bounding_box = functools.partial(make_bounding_box, format=features.BoundingBoxFormat.XYXY)


def make_bounding_boxes(
    formats=(features.BoundingBoxFormat.XYXY, features.BoundingBoxFormat.XYWH, features.BoundingBoxFormat.CXCYWH),
    image_sizes=((32, 32),),
    dtypes=(torch.int64, torch.float32),
    extra_dims=((4,), (2, 3)),
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
    extra_dims=((4,), (2, 3)),
):
    for num_categories_ in num_categories:
        yield make_one_hot_label(categories=[f"category{idx}" for idx in range(num_categories_)])

    for extra_dims_ in extra_dims:
        yield make_one_hot_label(extra_dims_)


class SampleInput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FunctionalInfo:
    def __init__(self, name, *, sample_inputs_fn):
        self.name = name
        self.functional = getattr(F, name)
        self._sample_inputs_fn = sample_inputs_fn

    def sample_inputs(self):
        yield from self._sample_inputs_fn()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], SampleInput):
            sample_input = args[0]
            return self.functional(*sample_input.args, **sample_input.kwargs)

        return self.functional(*args, **kwargs)


FUNCTIONAL_INFOS = []


def register_kernel_info_from_sample_inputs_fn(sample_inputs_fn):
    FUNCTIONAL_INFOS.append(FunctionalInfo(sample_inputs_fn.__name__, sample_inputs_fn=sample_inputs_fn))
    return sample_inputs_fn


@register_kernel_info_from_sample_inputs_fn
def horizontal_flip_image_tensor():
    for image in make_images():
        yield SampleInput(image)


@register_kernel_info_from_sample_inputs_fn
def horizontal_flip_bounding_box():
    for bounding_box in make_bounding_boxes(formats=[features.BoundingBoxFormat.XYXY]):
        yield SampleInput(bounding_box, format=bounding_box.format, image_size=bounding_box.image_size)


@register_kernel_info_from_sample_inputs_fn
def resize_image_tensor():
    for image, interpolation in itertools.product(
        make_images(),
        [
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.NEAREST,
        ],
    ):
        height, width = image.shape[-2:]
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield SampleInput(image, size=size, interpolation=interpolation)


@register_kernel_info_from_sample_inputs_fn
def resize_bounding_box():
    for bounding_box in make_bounding_boxes():
        height, width = bounding_box.image_size
        for size in [
            (height, width),
            (int(height * 0.75), int(width * 1.25)),
        ]:
            yield SampleInput(bounding_box, size=size, image_size=bounding_box.image_size)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(kernel, id=name)
        for name, kernel in F.__dict__.items()
        if not name.startswith("_")
        and callable(kernel)
        and any(feature_type in name for feature_type in {"image", "segmentation_mask", "bounding_box", "label"})
        and "pil" not in name
    ],
)
def test_scriptable(kernel):
    jit.script(kernel)


@pytest.mark.parametrize(
    ("functional_info", "sample_input"),
    [
        pytest.param(functional_info, sample_input, id=f"{functional_info.name}-{idx}")
        for functional_info in FUNCTIONAL_INFOS
        for idx, sample_input in enumerate(functional_info.sample_inputs())
    ],
)
def test_eager_vs_scripted(functional_info, sample_input):
    eager = functional_info(sample_input)
    scripted = jit.script(functional_info.functional)(*sample_input.args, **sample_input.kwargs)

    torch.testing.assert_close(eager, scripted)
