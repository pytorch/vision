import math
import torch

from enum import Enum
from torch import Tensor
from torch.jit.annotations import List, Tuple
from typing import Optional

from . import functional as F, InterpolationMode


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    """
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


def _shearX(img: Tensor, magnitude: float, fill: Optional[List[float]]) -> Tensor:
    v = math.degrees(magnitude)
    return F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[v, 0.0],
                    interpolation=InterpolationMode.BICUBIC, fill=fill)


def _shearY(img: Tensor, magnitude: float, fill: Optional[List[float]]) -> Tensor:
    v = math.degrees(magnitude)
    return F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, v],
                    interpolation=InterpolationMode.BICUBIC, fill=fill)


def _translateX(img: Tensor, magnitude: float, fill: Optional[List[float]]) -> Tensor:
    v = int(F._get_image_size(img)[0] * magnitude)
    return F.affine(img, angle=0.0, translate=[v, 0], scale=1.0, shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BICUBIC, fill=fill)


def _translateY(img: Tensor, magnitude: float, fill: Optional[List[float]]) -> Tensor:
    v = int(F._get_image_size(img)[1] * magnitude)
    return F.affine(img, angle=0.0, translate=[0, v], scale=1.0, shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BICUBIC, fill=fill)


def _rotate(img: Tensor, magnitude: float, fill: Optional[List[float]]) -> Tensor:
    return F.rotate(img, magnitude, interpolation=InterpolationMode.BICUBIC, fill=fill)


def _brightness(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    v = 1.0 + magnitude
    return F.adjust_brightness(img, v)


def _color(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    v = 1.0 + magnitude
    return F.adjust_saturation(img, v)


def _contrast(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    v = 1.0 + magnitude
    return F.adjust_contrast(img, v)


def _sharpness(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    v = 1.0 + magnitude
    return F.adjust_sharpness(img, v)


def _posterize(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    v = int(magnitude)
    return F.posterize(img, v)


def _solarize(img: Tensor, magnitude: float, _: Optional[List[float]]) -> Tensor:
    return F.solarize(img, magnitude)


def _autocontrast(img: Tensor, _: float, __: Optional[List[float]]) -> Tensor:
    return F.autocontrast(img)


def _equalize(img: Tensor, _: float, __: Optional[List[float]]) -> Tensor:
    return F.equalize(img)


def _invert(img: Tensor, _: float, __: Optional[List[float]]) -> Tensor:
    return F.invert(img)


_BINS = 10

_OPERATIONS = {
    # name: (method, magnitudes, signed)
    "ShearX": (_shearX, torch.linspace(0.0, 0.3, _BINS), True),
    "ShearY": (_shearY, torch.linspace(0.0, 0.3, _BINS), True),
    "TranslateX": (_translateX, torch.linspace(0.0, 150.0 / 331.0, _BINS), True),
    "TranslateY": (_translateY, torch.linspace(0.0, 150.0 / 331.0, _BINS), True),
    "Rotate": (_rotate, torch.linspace(0.0, 30.0, _BINS), True),
    "Brightness": (_brightness, torch.linspace(0.0, 0.9, _BINS), True),
    "Color": (_color, torch.linspace(0.0, 0.9, _BINS), True),
    "Contrast": (_contrast, torch.linspace(0.0, 0.9, _BINS), True),
    "Sharpness": (_sharpness, torch.linspace(0.0, 0.9, _BINS), True),
    "Posterize": (_posterize, torch.tensor([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]), False),
    "Solarize": (_solarize, torch.linspace(256.0, 0.0, _BINS), False),
    "AutoContrast": (_autocontrast, None, None),
    "Equalize": (_equalize, None, None),
    "Invert": (_invert, None, None),
}

_POLICIES = {
    AutoAugmentPolicy.IMAGENET: [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
        (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
        (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    ],
    AutoAugmentPolicy.CIFAR10: [
        (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
        (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
        (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
        (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
        (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
        (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
        (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
        (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
        (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
        (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
        (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
        (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
        (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
        (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
        (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
        (("Color", 0.9, 9), ("Equalize", 0.6, None)),
        (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
        (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
        (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
        (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
        (("Equalize", 0.8, None), ("Invert", 0.1, None)),
        (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
    ],
    AutoAugmentPolicy.SVHN: [
        (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
        (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
        (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
        (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
        (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
        (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
        (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
        (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
        (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
        (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
        (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
        (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
        (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
        (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
        (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
        (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
        (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
        (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
        (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
    ],
}


class AutoAugment(torch.nn.Module):
    r"""AutoAugment method, based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    """
    def __init__(self, policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET, fill: Optional[List[float]] = None):
        super().__init__()
        self.policy = policy
        self.fill = fill
        if policy not in _POLICIES:
            raise ValueError("The provided policy {} is not recognized.".format(policy))
        self.policies = _POLICIES[policy]

    @staticmethod
    def get_params(policy_num: int) -> Tuple[int, Tensor, Tensor]:
        policy_id = torch.randint(policy_num, (1,)).item()
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img):
        policy_id, probs, signs = self.get_params(len(self.policies))

        for i, (name, p, magnitude_id) in enumerate(self.policy[policy_id]):
            if probs[i] <= p:
                method, magnitudes, signed = _OPERATIONS[name]
                magnitude = magnitudes[magnitude_id] if magnitudes is not None else None
                if signed and signs[i] == 0:
                    magnitude *= -1
                img = method(img, magnitude, self.fill)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(policy={},fill={})'.format(self.policy, self.fill)
