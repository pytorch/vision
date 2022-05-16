import math
import numbers
import warnings
from typing import Any, Dict, Tuple

import torch
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F, Transform

from ._transform import _RandomApplyTransform
from ._utils import get_image_dimensions, has_all, has_any, is_simple_tensor, query_image


class RandomErasing(_RandomApplyTransform):
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
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
        self.value = value

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        image = query_image(sample)
        img_c, img_h, img_w = get_image_dimensions(image)

        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img_c)):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of input channels)"
            )

        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(self.ratio))
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

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break
        else:
            i, j, h, w, v = 0, 0, img_h, img_w, image

        return dict(zip("ijhwv", (i, j, h, w, v)))

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.erase_image_tensor(input, **params)
            return features.Image.new_like(input, output)
        elif is_simple_tensor(input):
            return F.erase_image_tensor(input, **params)
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")

        return super().forward(sample)


class RandomMixup(Transform):
    def __init__(self, *, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.mixup_image_tensor(input, **params)
            return features.Image.new_like(input, output)
        elif isinstance(input, features.OneHotLabel):
            output = F.mixup_one_hot_label(input, **params)
            return features.OneHotLabel.new_like(input, output)
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        elif not has_all(sample, features.Image, features.OneHotLabel):
            raise TypeError(f"{type(self).__name__}() is only defined for Image's *and* OneHotLabel's.")
        return super().forward(sample)


class RandomCutmix(Transform):
    def __init__(self, *, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))

        image = query_image(sample)
        _, H, W = get_image_dimensions(image)

        r_x = torch.randint(W, ())
        r_y = torch.randint(H, ())

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if isinstance(input, features.Image):
            output = F.cutmix_image_tensor(input, box=params["box"])
            return features.Image.new_like(input, output)
        elif isinstance(input, features.OneHotLabel):
            output = F.cutmix_one_hot_label(input, lam_adjusted=params["lam_adjusted"])
            return features.OneHotLabel.new_like(input, output)
        else:
            return input

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if has_any(sample, features.BoundingBox, features.SegmentationMask):
            raise TypeError(f"BoundingBox'es and SegmentationMask's are not supported by {type(self).__name__}()")
        elif not has_all(sample, features.Image, features.OneHotLabel):
            raise TypeError(f"{type(self).__name__}() is only defined for Image's *and* OneHotLabel's.")
        return super().forward(sample)
