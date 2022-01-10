from typing import Any, Optional, Dict, List

import torch

from ._transform import Transform


class _ContainerTransform(Transform):
    def _make_repr(self, lines: List[str]) -> str:
        extra_repr = self.extra_repr()
        if extra_repr:
            lines = [self.extra_repr(), *lines]
        head = f"{type(self).__name__}("
        tail = ")"
        body = [f"  {line.rstrip()}" for line in lines]
        return "\n".join([head, *body, tail])


class _WrapperTransform(_ContainerTransform):
    def __init__(self, transform: Transform):
        super().__init__()
        self._transform = transform

    def __repr__(self) -> str:
        return self._make_repr(repr(self._transform).splitlines())


class _MultiTransform(_ContainerTransform):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self._transforms = transforms

    def __repr__(self) -> str:
        lines = []
        for idx, transform in enumerate(self._transforms):
            partial_lines = repr(transform).splitlines()
            lines.append(f"({idx:d}): {partial_lines[0]}")
            lines.extend(partial_lines[1:])
        return self._make_repr(lines)


class Compose(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self._transforms:
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

        return self._transform(sample, params=params)

    def extra_repr(self) -> str:
        return f"p={self._p}"


class RandomChoice(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        idx = int(torch.randint(len(self._transforms), size=()))
        transform = self._transforms[idx]
        return transform(*inputs)


class RandomOrder(_MultiTransform):
    def forward(self, *inputs: Any) -> Any:  # type: ignore[override]
        for idx in torch.randperm(len(self._transforms)):
            transform = self._transforms[idx]
            inputs = transform(*inputs)
        return inputs
