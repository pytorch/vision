from typing import Any, Optional, List

import torch

from ._transform import Transform


class Compose(Transform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms
        for idx, transform in enumerate(transforms):
            self.add_module(str(idx), transform)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomApply(Transform):
    def __init__(self, transform: Transform, *, p: float = 0.5) -> None:
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if float(torch.rand(())) < self.p:
            return sample

        return self.transform(sample)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class RandomChoice(Transform):
    def __init__(self, *transforms: Transform, probabilities: Optional[List[float]] = None) -> None:
        if probabilities is None:
            probabilities = [1] * len(transforms)
        elif len(probabilities) != len(transforms):
            raise ValueError(
                f"The number of probabilities doesn't match the number of transforms: "
                f"{len(probabilities)} != {len(transforms)}"
            )

        super().__init__()

        self.transforms = transforms
        for idx, transform in enumerate(transforms):
            self.add_module(str(idx), transform)

        total = sum(probabilities)
        self.probabilities = [p / total for p in probabilities]

    def forward(self, *inputs: Any) -> Any:
        idx = int(torch.multinomial(torch.tensor(self.probabilities), 1))
        transform = self.transforms[idx]
        return transform(*inputs)


class RandomOrder(Transform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms
        for idx, transform in enumerate(transforms):
            self.add_module(str(idx), transform)

    def forward(self, *inputs: Any) -> Any:
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            inputs = transform(*inputs)
        return inputs
