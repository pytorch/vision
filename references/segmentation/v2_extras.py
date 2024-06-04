"""This file only exists to be lazy-imported and avoid V2-related import warnings when just using V1."""
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class PadIfSmaller(v2.Transform):
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = v2._utils._setup_fill_arg(fill)

    def _get_params(self, sample):
        _, height, width = v2._utils.query_chw(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = v2._utils._get_fill(self.fill, type(inpt))
        fill = v2._utils._convert_fill_arg(fill)

        return v2.functional.pad(inpt, padding=params["padding"], fill=fill)


class CocoDetectionToVOCSegmentation(v2.Transform):
    """Turn samples from datasets.CocoDetection into the same format as VOCSegmentation.

    This is achieved in two steps:

    1. COCO differentiates between 91 categories while VOC only supports 21, including background for both. Fortunately,
       the COCO categories are a superset of the VOC ones and thus can be mapped. Instances of the 70 categories not
       present in VOC are dropped and replaced by background.
    2. COCO only offers detection masks, i.e. a (N, H, W) bool-ish tensor, where the truthy values in each individual
       mask denote the instance. However, a segmentation mask is a (H, W) integer tensor (typically torch.uint8), where
       the value of each pixel denotes the category it belongs to. The detection masks are merged into one segmentation
       mask while pixels that belong to multiple detection masks are marked as invalid.
    """

    COCO_TO_VOC_LABEL_MAP = dict(
        zip(
            [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72],
            range(21),
        )
    )
    INVALID_VALUE = 255

    def _coco_detection_masks_to_voc_segmentation_mask(self, target):
        if "masks" not in target:
            return None

        instance_masks, instance_labels_coco = target["masks"], target["labels"]

        valid_labels_voc = [
            (idx, label_voc)
            for idx, label_coco in enumerate(instance_labels_coco.tolist())
            if (label_voc := self.COCO_TO_VOC_LABEL_MAP.get(label_coco)) is not None
        ]

        if not valid_labels_voc:
            return None

        valid_voc_category_idcs, instance_labels_voc = zip(*valid_labels_voc)

        instance_masks = instance_masks[list(valid_voc_category_idcs)].to(torch.uint8)
        instance_labels_voc = torch.tensor(instance_labels_voc, dtype=torch.uint8)

        # Calling `.max()` on the stacked detection masks works fine to separate background from foreground as long as
        # there is at most a single instance per pixel. Overlapping instances will be filtered out in the next step.
        segmentation_mask, _ = (instance_masks * instance_labels_voc.reshape(-1, 1, 1)).max(dim=0)
        segmentation_mask[instance_masks.sum(dim=0) > 1] = self.INVALID_VALUE

        return segmentation_mask

    def forward(self, image, target):
        segmentation_mask = self._coco_detection_masks_to_voc_segmentation_mask(target)
        if segmentation_mask is None:
            segmentation_mask = torch.zeros(v2.functional.get_size(image), dtype=torch.uint8)

        return image, tv_tensors.Mask(segmentation_mask)
