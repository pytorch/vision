from __future__ import annotations

from typing import Any, Tuple, Optional, Sequence, Union, Collection

import torch
from torchvision.prototype.utils._internal import StrEnum

from ._feature import _Feature


class KeypointSymmetry(StrEnum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class Keypoint(_Feature):
    image_size: Tuple[int, int]
    descriptions: Sequence[Sequence[str]]
    symmetries: Sequence[Tuple[KeypointSymmetry, int, int]]

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int],
        descriptions: Optional[Sequence[str]] = None,
        symmetries: Collection[
            Tuple[Union[str, KeypointSymmetry], Union[str, int], Union[str, int]],
        ] = (),
    ) -> Keypoint:
        keypoint = super().__new__(cls, data, dtype=dtype, device=device)

        parsed_symmetries = []
        for symmetry, first, second in symmetries:
            if isinstance(symmetry, str):
                symmetry = KeypointSymmetry[symmetry]

            if isinstance(first, str):
                if not descriptions:
                    raise ValueError

                first = descriptions.index(first)

            if isinstance(second, str):
                if not descriptions:
                    raise ValueError

                second = descriptions.index(second)

            parsed_symmetries.append((symmetry, first, second))

        keypoint._metadata.update(dict(image_size=image_size, descriptions=descriptions, symmetries=parsed_symmetries))

        return keypoint

    @classmethod
    def _to_tensor(cls, data: Any, *, dtype: Optional[torch.dtype], device: Optional[torch.device]) -> torch.Tensor:
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if tensor.shape[-1] != 2:
            raise ValueError
        if tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        return tensor

    @property
    def num_keypoints(self) -> int:
        return self.shape[-2]
