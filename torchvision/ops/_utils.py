from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert (
                _tensor.size(1) == 4
            ), "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
    elif isinstance(boxes, torch.Tensor):
        assert boxes.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        assert False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
    return


def split_normalization_params(
    model: nn.Module, norm_classes: Optional[List[type]] = None
) -> Tuple[List[Tensor], List[Tensor]]:
    # Adapted from https://github.com/facebookresearch/ClassyVision/blob/659d7f78/classy_vision/generic/util.py#L501
    if not norm_classes:
        norm_classes = [nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm]

    for t in norm_classes:
        if not issubclass(t, nn.Module):
            raise ValueError(f"Class {t} is not a subclass of nn.Module.")

    classes = tuple(norm_classes)

    norm_params = []
    other_params = []
    for module in model.modules():
        if next(module.children(), None):
            other_params.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
        elif isinstance(module, classes):
            norm_params.extend(p for p in module.parameters() if p.requires_grad)
        else:
            other_params.extend(p for p in module.parameters() if p.requires_grad)
    return norm_params, other_params
