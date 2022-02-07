import math
from typing import Any, Dict, TypeVar

import torch
from torchvision import transforms as _transforms
from torchvision.prototype import features
from torchvision.prototype.transforms import Transform

from . import functional as F
from .utils import Query

T = TypeVar("T", bound=features.Feature)


class RandomErasing(Transform):
    _LEGACY_TRANSFORM_CLS = _transforms.RandomErasing
    _DISPATCH = F.erase

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        legacy_transform = self._LEGACY_TRANSFORM_CLS(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
        # TODO: deprecate p in favor of wrapping the transform in a RandomApply
        self.p = legacy_transform.p
        self.scale = legacy_transform.scale
        self.ratio = legacy_transform.ratio
        self.value = legacy_transform.value
        self.inplace = legacy_transform.inplace

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

        i, j, h, w, v = self._LEGACY_TRANSFORM_CLS.get_params(image, scale=self.scale, ratio=self.ratio, value=value)
        return dict(i=i, j=j, h=h, w=w, v=v, inplace=self.inplace)

    def _dispatch(self, feature: T, params: Dict[str, Any]) -> T:
        if torch.rand(1) >= self.p:
            return feature

        return super()._dispatch(feature, params)

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("p", "scale", "ratio", "value", "inplace")


class RandomMixup(Transform):
    _DISPATCH = F.mixup

    def __init__(self, *, alpha: float, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.inplace = inplace

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())), inplace=self.inplace)

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("alpha", "inplace")


class RandomCutmix(Transform):
    _DISPATCH = F.cutmix

    def __init__(self, *, alpha: float, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.inplace = inplace

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

        return dict(box=box, lam_adjusted=lam_adjusted, inplace=self.inplace)

    def extra_repr(self) -> str:
        return self._extra_repr_from_attrs("alpha", "inplace")
