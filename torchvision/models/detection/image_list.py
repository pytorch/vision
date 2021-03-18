# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import Tensor
from typing import List, Tuple, NamedTuple


class ImageList(NamedTuple):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    tensors: Tensor
    image_sizes: List[Tuple[int, int]]

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
