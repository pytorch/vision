import PIL.Image
import pytest
import torch
from test_prototype_transforms_kernels import make_images, make_bounding_boxes
from torchvision.prototype import transforms, features
from torchvision.transforms.functional import to_pil_image


def make_vanilla_tensor_image(*args, **kwargs):
    for image in make_images(*args, **kwargs):
        if image.ndim > 3:
            continue
        yield image.data


def make_pil_image(*args, **kwargs):
    for image in make_vanilla_tensor_image(*args, **kwargs):
        yield to_pil_image(image)


INPUT_CREATIONS_FNS = {
    features.Image: make_images,
    features.BoundingBox: make_bounding_boxes,
    torch.Tensor: make_vanilla_tensor_image,
    PIL.Image.Image: make_pil_image,
}


def parametrize(*transforms):
    params = []
    for transform in transforms:
        dispatcher = transform._DISPATCHER
        if dispatcher is None:
            continue

        for type_ in dispatcher._kernels:
            try:
                inputs = INPUT_CREATIONS_FNS[type_]()
            except KeyError:
                continue

            params.extend(
                pytest.param(
                    transform,
                    input,
                    id=f"{type(transform).__name__}-{type_.__module__}.{type_.__name__}-{idx}",
                )
                for idx, input in enumerate(inputs)
            )

    return pytest.mark.parametrize(("transform", "input"), params)


@parametrize(
    transforms.RandomErasing(),
    transforms.RandomMixup(alpha=1.0),
    transforms.RandomCutmix(alpha=1.0),
    transforms.RandAugment(),
    transforms.TrivialAugmentWide(),
    transforms.AutoAugment(),
    transforms.HorizontalFlip(),
    transforms.Resize([16]),
    transforms.CenterCrop([16]),
    transforms.RandomResizedCrop([16]),
    transforms.ConvertBoundingBoxFormat("xyxy"),
    transforms.ConvertImageDtype(),
    transforms.ConvertColorSpace("grayscale"),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
)
def test_smoke(transform, input):
    transform(input)
