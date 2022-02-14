import math
from typing import Any, Dict, Tuple

import torch
from torchvision import transforms as _transforms
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform, kernels as K
from torchvision.transforms import functional as _F

from .utils import Query


class RandomErasing(Transform):
    _LEGACY_TRANSFORM_CLS = _transforms.RandomErasing

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
    ):
        super().__init__()
        legacy_transform = self._LEGACY_TRANSFORM_CLS(p=p, scale=scale, ratio=ratio, value=value, inplace=False)
        # TODO: deprecate p in favor of wrapping the transform in a RandomApply
        self.p = legacy_transform.p
        self.scale = legacy_transform.scale
        self.ratio = legacy_transform.ratio
        self.value = legacy_transform.value

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image = Query(sample).image_for_size_and_channels_extraction()

        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value

        if value is not None and not (len(value) in (1, image.shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                f"{image.shape[-3]} (number of input channels)"
            )

        return dict(
            zip("ijhwv", self._LEGACY_TRANSFORM_CLS.get_params(image, scale=self.scale, ratio=self.ratio, value=value))
        )

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if torch.rand(1) >= self.p:
            return input

        if type(input) is torch.Tensor:
            return _F.erase(input, **params)
        elif type(input) is features.Image:
            return features.Image.new_like(input, K.erase_image(input, **params))
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("p", "scale", "ratio", "value")


class RandomMixup(Transform):
    def __init__(self, *, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))

    def _supports(self, obj: Any) -> bool:
        return type(obj) in {features.Image, features.OneHotLabel}

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.mixup_image(input, **params)
            return features.Image.new_like(input, output)
        elif type(input) is features.OneHotLabel:
            output = K.mixup_one_hot_label(input, **params)
            return features.OneHotLabel.new_like(input, output)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("alpha")


class RandomCutmix(Transform):
    def __init__(self, *, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def get_params(self, sample: Any) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))

        H, W = Query(sample).image_size()

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

    def _supports(self, obj: Any) -> bool:
        return type(obj) in {features.Image, features.OneHotLabel}

    def _transform(self, input: Any, params: Dict[str, Any]) -> Any:
        if type(input) is features.Image:
            output = K.cutmix_image(input, box=params["box"])
            return features.Image.new_like(input, output)
        elif type(input) is features.OneHotLabel:
            output = K.cutmix_one_hot_label(input, lam_adjusted=params["lam_adjusted"])
            return features.OneHotLabel.new_like(input, output)
        else:
            return input

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("alpha")
