import torch
from torch import Tensor
from torch.jit.annotations import List


def _cat(tensors, dim=0):
    # type: (List[Tensor], int) -> Tensor
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes):
    # type: (List[Tensor]) -> Tensor
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert _tensor.size(1) == 4, \
                'The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]'
    elif isinstance(boxes, torch.Tensor):
        assert boxes.size(1) == 5, 'The boxes tensor shape is not correct as Tensor[K, 5]'
    else:
        assert False, 'boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]'
    return
