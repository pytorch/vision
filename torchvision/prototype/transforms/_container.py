from typing import Any

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
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms
        for idx, transform in enumerate(transforms):
            self.add_module(str(idx), transform)

    def forward(self, *inputs: Any) -> Any:
        idx = int(torch.randint(len(self.transforms), size=()))
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
