import importlib.machinery
import importlib.util
import random
from pathlib import Path

import pytest

import torch
import torchvision.transforms.v2 as v2_transforms
from common_utils import assert_equal
from torchvision import tv_tensors

from torchvision.transforms import functional as legacy_F
from torchvision.transforms.v2 import functional as prototype_F
from torchvision.transforms.v2._utils import _get_fill, query_size
from torchvision.transforms.v2.functional import to_pil_image

from transforms_v2_legacy_utils import make_bounding_boxes, make_detection_mask, make_image, make_segmentation_mask


def import_transforms_from_references(reference):
    HERE = Path(__file__).parent
    PROJECT_ROOT = HERE.parent

    loader = importlib.machinery.SourceFileLoader(
        "transforms", str(PROJECT_ROOT / "references" / reference / "transforms.py")
    )
    spec = importlib.util.spec_from_loader("transforms", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


det_transforms = import_transforms_from_references("detection")


class TestRefDetTransforms:
    def make_tv_tensors(self, with_mask=True):
        size = (600, 800)
        num_objects = 22

        def make_label(extra_dims, categories):
            return torch.randint(categories, extra_dims, dtype=torch.int64)

        pil_image = to_pil_image(make_image(size=size, color_space="RGB"))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (pil_image, target)

        tensor_image = torch.Tensor(make_image(size=size, color_space="RGB", dtype=torch.float32))
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (tensor_image, target)

        tv_tensor_image = make_image(size=size, color_space="RGB", dtype=torch.float32)
        target = {
            "boxes": make_bounding_boxes(canvas_size=size, format="XYXY", batch_dims=(num_objects,), dtype=torch.float),
            "labels": make_label(extra_dims=(num_objects,), categories=80),
        }
        if with_mask:
            target["masks"] = make_detection_mask(size=size, num_objects=num_objects, dtype=torch.long)

        yield (tv_tensor_image, target)

    @pytest.mark.parametrize(
        "t_ref, t, data_kwargs",
        [
            (det_transforms.RandomHorizontalFlip(p=1.0), v2_transforms.RandomHorizontalFlip(p=1.0), {}),
            (
                det_transforms.RandomIoUCrop(),
                v2_transforms.Compose(
                    [
                        v2_transforms.RandomIoUCrop(),
                        v2_transforms.SanitizeBoundingBoxes(labels_getter=lambda sample: sample[1]["labels"]),
                    ]
                ),
                {"with_mask": False},
            ),
            (det_transforms.RandomZoomOut(), v2_transforms.RandomZoomOut(), {"with_mask": False}),
            (det_transforms.ScaleJitter((1024, 1024)), v2_transforms.ScaleJitter((1024, 1024), antialias=True), {}),
            (
                det_transforms.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                ),
                v2_transforms.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                ),
                {},
            ),
        ],
    )
    def test_transform(self, t_ref, t, data_kwargs):
        for dp in self.make_tv_tensors(**data_kwargs):

            # We should use prototype transform first as reference transform performs inplace target update
            torch.manual_seed(12)
            output = t(dp)

            torch.manual_seed(12)
            expected_output = t_ref(*dp)

            assert_equal(expected_output, output)


seg_transforms = import_transforms_from_references("segmentation")


# We need this transform for two reasons:
# 1. transforms.RandomCrop uses a different scheme to pad images and masks of insufficient size than its name
#    counterpart in the detection references. Thus, we cannot use it with `pad_if_needed=True`
# 2. transforms.Pad only supports a fixed padding, but the segmentation datasets don't have a fixed image size.
class PadIfSmaller(v2_transforms.Transform):
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = v2_transforms._geometry._setup_fill_arg(fill)

    def _get_params(self, sample):
        height, width = query_size(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = _get_fill(self.fill, type(inpt))
        return prototype_F.pad(inpt, padding=params["padding"], fill=fill)


class TestRefSegTransforms:
    def make_tv_tensors(self, supports_pil=True, image_dtype=torch.uint8):
        size = (256, 460)
        num_categories = 21

        conv_fns = []
        if supports_pil:
            conv_fns.append(to_pil_image)
        conv_fns.extend([torch.Tensor, lambda x: x])

        for conv_fn in conv_fns:
            tv_tensor_image = make_image(size=size, color_space="RGB", dtype=image_dtype)
            tv_tensor_mask = make_segmentation_mask(size=size, num_categories=num_categories, dtype=torch.uint8)

            dp = (conv_fn(tv_tensor_image), tv_tensor_mask)
            dp_ref = (
                to_pil_image(tv_tensor_image) if supports_pil else tv_tensor_image.as_subclass(torch.Tensor),
                to_pil_image(tv_tensor_mask),
            )

            yield dp, dp_ref

    def set_seed(self, seed=12):
        torch.manual_seed(seed)
        random.seed(seed)

    def check(self, t, t_ref, data_kwargs=None):
        for dp, dp_ref in self.make_tv_tensors(**data_kwargs or dict()):

            self.set_seed()
            actual = actual_image, actual_mask = t(dp)

            self.set_seed()
            expected_image, expected_mask = t_ref(*dp_ref)
            if isinstance(actual_image, torch.Tensor) and not isinstance(expected_image, torch.Tensor):
                expected_image = legacy_F.pil_to_tensor(expected_image)
            expected_mask = legacy_F.pil_to_tensor(expected_mask).squeeze(0)
            expected = (expected_image, expected_mask)

            assert_equal(actual, expected)

    @pytest.mark.parametrize(
        ("t_ref", "t", "data_kwargs"),
        [
            (
                seg_transforms.RandomHorizontalFlip(flip_prob=1.0),
                v2_transforms.RandomHorizontalFlip(p=1.0),
                dict(),
            ),
            (
                seg_transforms.RandomHorizontalFlip(flip_prob=0.0),
                v2_transforms.RandomHorizontalFlip(p=0.0),
                dict(),
            ),
            (
                seg_transforms.RandomCrop(size=480),
                v2_transforms.Compose(
                    [
                        PadIfSmaller(size=480, fill={tv_tensors.Mask: 255, "others": 0}),
                        v2_transforms.RandomCrop(size=480),
                    ]
                ),
                dict(),
            ),
            (
                seg_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                v2_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(supports_pil=False, image_dtype=torch.float),
            ),
        ],
    )
    def test_common(self, t_ref, t, data_kwargs):
        self.check(t, t_ref, data_kwargs)
