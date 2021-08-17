import math
import torch

from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict

from . import functional as F, InterpolationMode

__all__ = ["AutoAugmentPolicy", "AutoAugment", "TrivialAugment"]


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


def _get_transforms(  # type: ignore[return]
    policy: AutoAugmentPolicy
) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
    if policy == AutoAugmentPolicy.IMAGENET:
        return [
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
        ]
    elif policy == AutoAugmentPolicy.CIFAR10:
        return [
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
        ]
    elif policy == AutoAugmentPolicy.SVHN:
        return [
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
        ]


def _get_magnitudes(
        augmentation_space: str, image_size: List[int], num_bins: int = 10
) -> Dict[str, Tuple[Tensor, bool]]:
    if augmentation_space == 'aa':
        shear_max = 0.3
        translate_max_x = 150.0 / 331.0 * image_size[0]
        translate_max_y = 150.0 / 331.0 * image_size[1]
        rotate_max = 30.0
        enhancer_max = 0.9
        posterize_min_bits = 4

    elif augmentation_space == 'ta_wide':
        shear_max = 0.99
        translate_max_x = 32.0  # this is an absolute
        translate_max_y = 32.0  # this is an absolute
        rotate_max = 135.0
        enhancer_max = 0.99
        posterize_min_bits = 2
    else:
        raise ValueError(f"Provided augmentation_space arguments {augmentation_space} not available.")

    magnitudes = {
        # name: (magnitudes, signed)
        "ShearX": (torch.linspace(0.0, shear_max, num_bins), True),
        "ShearY": (torch.linspace(0.0, shear_max, num_bins), True),
        "TranslateX": (torch.linspace(0.0, translate_max_x, num_bins), True),
        "TranslateY": (torch.linspace(0.0, translate_max_y, num_bins), True),
        "Rotate": (torch.linspace(0.0, rotate_max, num_bins), True),
        "Brightness": (torch.linspace(0.0, enhancer_max, num_bins), True),
        "Color": (torch.linspace(0.0, enhancer_max, num_bins), True),
        "Contrast": (torch.linspace(0.0, enhancer_max, num_bins), True),
        "Sharpness": (torch.linspace(0.0, enhancer_max, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / (8 - posterize_min_bits))).round().int(), False),
        "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(float('nan')), False),
        "Equalize": (torch.tensor(float('nan')), False),
        "Invert": (torch.tensor(float('nan')), False),
    }
    return magnitudes


def apply_aug(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class TrivialAugment(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
        If the image is torch Tensor, it should be of type torch.uint8, and it is expected
        to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
        If img is PIL Image, it is expected to be in mode "L" or "RGB".

        Args:
            augmentation_space (str): A string defining which augmentation space to use.
                The augmentation space can either set to be the one used for AutoAugment (`aa`)
                or to the strongest augmentation space from the TrivialAugment paper (`ta_wide`).
            num_magnitude_bins (int): The number of different magnitude values.
            interpolation (InterpolationMode): Desired interpolation enum defined by
                :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
                If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            fill (sequence or number, optional): Pixel fill value for the area outside the transformed
                image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, augmentation_space: str = 'ta_wide', num_magnitude_bins: int = 30,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None):
        super().__init__()
        self.augmentation_space = augmentation_space
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, img: Tensor):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = _get_magnitudes(self.augmentation_space, F._get_image_size(img), num_bins=self.num_magnitude_bins)
        op_index: int = torch.randint(len(op_meta), (1,)).item() # type: ignore[assignment]
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if not magnitudes.isnan().all() else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return apply_aug(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


class AutoAugment(torch.nn.Module):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill

        self.transforms = _get_transforms(policy)
        if self.transforms is None:
            raise ValueError("The provided policy {} is not recognized.".format(policy))

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.transforms))

        for i, (op_name, p, magnitude_id) in enumerate(self.transforms[transform_id]):
            if probs[i] <= p:
                op_meta = _get_magnitudes('aa', F._get_image_size(img))
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) \
                    if not magnitudes.isnan().all() and magnitude_id is not None else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = apply_aug(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(policy={}, fill={})'.format(self.policy, self.fill)
