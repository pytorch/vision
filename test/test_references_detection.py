import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "references" / "detection"))

import coco_utils

from coco_utils import ConvertCocoPolysToMask


def test_convert_coco_polys_to_mask_skips_masks_when_not_requested(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("mask decoding should be skipped")

    monkeypatch.setattr(coco_utils, "convert_coco_poly_to_mask", fail_if_called)

    _, target = ConvertCocoPolysToMask()(
        Image.new("RGB", (4, 4)),
        {
            "image_id": 1,
            "annotations": [
                {
                    "bbox": [0, 0, 2, 2],
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": 4,
                    "segmentation": [],
                }
            ],
        },
    )

    assert "masks" not in target
