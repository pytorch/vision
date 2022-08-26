import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torchvision.prototype.transforms import Transform

from ._transform import _RandomApplyTransform


class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        for transform in self.transforms:
            inputs = transform(*inputs)
        return inputs


class RandomApply(_RandomApplyTransform):
    def __init__(self, transform: Transform, *, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.transform = transform

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.transform(inpt)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class RandomChoice(Transform):
    def __init__(
        self,
        transforms: Sequence[Callable],
        probabilities: Optional[List[float]] = None,
        p: Optional[List[float]] = None,
    ) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is not None:
            warnings.warn(
                "Argument p is deprecated and will be removed in a future release. "
                "Please use probabilities argument instead."
            )
            probabilities = p

        if probabilities is None:
            probabilities = [1] * len(transforms)
        elif len(probabilities) != len(transforms):
            raise ValueError(
                f"The number of probabilities doesn't match the number of transforms: "
                f"{len(probabilities)} != {len(transforms)}"
            )

        super().__init__()

        self.transforms = transforms
        total = sum(probabilities)
        self.probabilities = [prob / total for prob in probabilities]

    def forward(self, *inputs: Any) -> Any:
        idx = int(torch.multinomial(torch.tensor(self.probabilities), 1))
        transform = self.transforms[idx]
        return transform(*inputs)


class RandomOrder(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        super().__init__()
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            inputs = transform(*inputs)
        return inputs
