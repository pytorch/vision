from typing import Dict, Any, Tuple, Optional, Sequence, Union, Collection

import torch
from torchvision.prototype.utils._internal import StrEnum

from ._feature import Feature, DEFAULT


class KeypointSymmetry(StrEnum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class Keypoint(Feature):
    image_size: Tuple[int, int]
    descriptions: Sequence[Sequence[str]]
    symmetries: Sequence[Tuple[KeypointSymmetry, Tuple[int, int]]]

    @classmethod
    def _to_tensor(cls, data, *, dtype, device):
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if tensor.shape[-1] != 2:
            raise ValueError
        if tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        return tensor

    @classmethod
    def _parse_meta_data(
        cls,
        image_size: Tuple[int, int] = DEFAULT,  # type: ignore[assignment]
        descriptions: Optional[Sequence[str]] = DEFAULT,  # type: ignore[assignment]
        symmetries: Collection[
            Tuple[Union[str, KeypointSymmetry], Union[str, int], Union[str, int]],
        ] = DEFAULT,
    ) -> Dict[str, Tuple[Any, Any]]:
        if symmetries is not DEFAULT:
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

            symmetries = parsed_symmetries

        return dict(
            image_size=(image_size, None),
            descriptions=(descriptions, None),
            symmetries=(symmetries, []),
        )

    @property
    def num_keypoints(self) -> int:
        return self.shape[-2]
