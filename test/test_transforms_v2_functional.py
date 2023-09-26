import numpy as np
import PIL.Image
import pytest
import torch

from torchvision.transforms.v2 import functional as F


@pytest.mark.parametrize(
    ("alias", "target"),
    [
        pytest.param(alias, target, id=alias.__name__)
        for alias, target in [
            (F.hflip, F.horizontal_flip),
            (F.vflip, F.vertical_flip),
            (F.get_image_num_channels, F.get_num_channels),
            (F.to_pil_image, F.to_pil_image),
            (F.elastic_transform, F.elastic),
            (F.to_grayscale, F.rgb_to_grayscale),
        ]
    ],
)
def test_alias(alias, target):
    assert alias is target


@pytest.mark.parametrize(
    "inpt",
    [
        127 * np.ones((32, 32, 3), dtype="uint8"),
        PIL.Image.new("RGB", (32, 32), 122),
    ],
)
def test_to_image(inpt):
    output = F.to_image(inpt)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)

    assert np.asarray(inpt).sum() == output.sum().item()


@pytest.mark.parametrize(
    "inpt",
    [
        torch.randint(0, 256, size=(3, 32, 32), dtype=torch.uint8),
        127 * np.ones((32, 32, 3), dtype="uint8"),
    ],
)
@pytest.mark.parametrize("mode", [None, "RGB"])
def test_to_pil_image(inpt, mode):
    output = F.to_pil_image(inpt, mode=mode)
    assert isinstance(output, PIL.Image.Image)

    assert np.asarray(inpt).sum() == np.asarray(output).sum()
