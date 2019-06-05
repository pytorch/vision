import torch


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes):
    concat_boxes = _cat([b for b in boxes], dim=0)
    ids = _cat(
        [
            torch.full_like(b[:, :1], i)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois
