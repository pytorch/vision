import torch


def convert_boxes_to_roi_format(boxes):
    concat_boxes = cat([b for b in boxes], dim=0)
    ids = cat(
	[
            torch.full_like(b[:, :1], i)
	    for i, b in enumerate(boxes)
	],
	dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1) 
    return rois
