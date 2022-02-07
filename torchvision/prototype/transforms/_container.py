from typing import Any, Optional, Dict

import torch

from ._transform import Transform


class _WrapperTransform(Transform):
    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform


class _MultiTransform(Transform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms
        # to properly register them for repr
        for idx, transform in enumerate(transforms):
            self.add_module(str(idx), transform)


class Compose(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomApply(_WrapperTransform):
    def __init__(self, transform: Transform, *, p: float = 0.5) -> None:
        super().__init__(transform)
        self._p = p

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if float(torch.rand(())) < self._p:
            return sample

        return self.transform(sample, params=params)

    def extra_repr(self) -> str:
        return f"p={self._p}"


class RandomChoice(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        idx = int(torch.randint(len(self.transforms), size=()))
        transform = self.transforms[idx]
        return transform(*inputs)


class RandomOrder(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            inputs = transform(*inputs)
        return inputs
