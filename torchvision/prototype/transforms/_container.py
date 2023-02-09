import warnings
from typing import Any, Callable, List, Optional, Sequence, Union

import torch

from torch import nn
from torchvision.prototype.transforms import Transform


class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


class RandomApply(Transform):
    def __init__(self, transforms: Union[Sequence[Callable], nn.ModuleList], p: float = 0.5) -> None:
        super().__init__()

        if not isinstance(transforms, (Sequence, nn.ModuleList)):
            raise TypeError("Argument transforms should be a sequence of callables or a `nn.ModuleList`")
        self.transforms = transforms

        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        if torch.rand(1) >= self.p:
            return sample

        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


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
        sample = inputs if len(inputs) > 1 else inputs[0]
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            sample = transform(sample)
        return sample
