import math
import numbers
import warnings
from typing import Any, Dict, List, Tuple, Union

import PIL.Image
import torch
from torchvision import datapoints, transforms as _transforms
from torchvision.transforms.v2 import functional as F

from ._transform import _RandomApplyTransform
from .utils import is_simple_tensor, query_chw


class RandomErasing(_RandomApplyTransform):
    """[BETA] Randomly select a rectangle region in the input image or video and erase its pixels.

    .. v2betastatus:: RandomErasing transform

    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
        p (float, optional): probability that the random erasing operation will be performed.
        scale (tuple of float, optional): range of proportion of erased area against input image.
        ratio (tuple of float, optional): range of aspect ratio of erased area.
        value (number or tuple of numbers): erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
        inplace (bool, optional): boolean to make this transform inplace. Default set to False.

    Returns:
        Erased input.

    Example:
        >>> from torchvision.transforms import v2 as transforms
        >>>
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    _v1_transform_cls = _transforms.RandomErasing

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        return dict(
            super()._extract_params_for_v1_transform(),
            value="random" if self.value is None else self.value,
        )

    _transformed_types = (is_simple_tensor, datapoints.Image, PIL.Image.Image, datapoints.Video)

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        self.scale = scale
        self.ratio = ratio
        if isinstance(value, (int, float)):
            self.value = [float(value)]
        elif isinstance(value, str):
            self.value = None
        elif isinstance(value, (list, tuple)):
            self.value = [float(v) for v in value]
        else:
            self.value = value
        self.inplace = inplace

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_c, img_h, img_w = query_chw(flat_inputs)

        if self.value is not None and not (len(self.value) in (1, img_c)):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of inpt channels)"
            )

        area = img_h * img_w

        log_ratio = self._log_ratio
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if self.value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(self.value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break
        else:
            i, j, h, w, v = 0, 0, img_h, img_w, None

        return dict(i=i, j=j, h=h, w=w, v=v)

    def _transform(
        self, inpt: Union[datapoints._ImageType, datapoints._VideoType], params: Dict[str, Any]
    ) -> Union[datapoints._ImageType, datapoints._VideoType]:
        if params["v"] is not None:
            inpt = F.erase(inpt, **params, inplace=self.inplace)

        return inpt
