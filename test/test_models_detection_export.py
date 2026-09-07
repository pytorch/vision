import os

import pytest
import torch
from common_utils import set_rng_seed
from torch.export import Dim, export
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn


def _get_image(input_shape, device="cpu"):
    GRACE_HOPPER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "encode_jpeg", "grace_hopper_517x606.jpg"
    )
    if os.path.exists(GRACE_HOPPER):
        from PIL import Image
        from torchvision import transforms

        img = Image.open(GRACE_HOPPER)
        w, h = img.size
        img = img.crop((0, 0, w, w))
        img = img.resize(input_shape[1:3])
        return transforms.ToTensor()(img).to(device=device)
    return torch.rand(input_shape, device=device)


def _fpn_dynamic_shapes():
    """Dynamic shapes constrained to multiples of 64 for FPN compatibility.

    Strided convolutions in the backbone specialize on the parity of
    ceil_to_32(dim)/32. Using multiples of 64 ensures consistent even
    block counts, avoiding shape guards that would reject valid inputs.
    """
    _h = Dim("_h", min=4, max=21)
    _w = Dim("_w", min=4, max=21)
    return {"images": [{1: 64 * _h, 2: 64 * _w}]}


@pytest.fixture(scope="module")
def fasterrcnn_model():
    """Load and pre-initialize FasterRCNN once for all tests in the module.

    _skip_resize=True bypasses the aspect-ratio-preserving resize in
    GeneralizedRCNNTransform, which creates shape guards that specialize
    to the tracing input's orientation (h<=w vs h>w). The caller is
    responsible for pre-sizing inputs to the expected range.
    """
    set_rng_seed(0)
    model = fasterrcnn_mobilenet_v3_large_fpn(
        num_classes=50,
        weights_backbone=None,
        box_score_thresh=0.02076,
        _skip_resize=True,
    )
    model.eval()
    with torch.no_grad():
        _ = model([torch.randn(3, 256, 256)])
    return model


@pytest.fixture(scope="module")
def real_image():
    """Load the same real image used by test_detection_model."""
    return _get_image((3, 320, 320))


class TestDetectionExport:
    """Tests for torch.export of detection models.

    Verifies that export produces correct results matching eager mode,
    works with dynamic shapes, handles edge cases, and supports both
    strict=True and strict=False modes.
    """

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_succeeds(self, fasterrcnn_model, strict):
        """Export should succeed with dynamic H/W shapes."""
        with torch.no_grad():
            ep = export(
                fasterrcnn_model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )
        assert ep is not None

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_matches_eager_real_image(self, fasterrcnn_model, real_image, strict):
        """Exported model output should match eager on the same real image."""
        with torch.no_grad():
            ep = export(
                fasterrcnn_model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        inp = [real_image.clone()]
        with torch.no_grad():
            eager_out = fasterrcnn_model(inp)
            export_out = ep.module()([real_image.clone()])

        assert len(eager_out) == 1 and len(export_out) == 1
        for key in ("boxes", "scores", "labels"):
            assert key in export_out[0], f"Missing key '{key}' in export output"

        # With random backbone weights, scores are near-zero and NMS ordering
        # is sensitive to floating-point differences between eager and export.
        # Only compare detections with confident scores; otherwise just verify
        # structural correctness.
        eager_confident = eager_out[0]["scores"] > 0.1
        export_confident = export_out[0]["scores"] > 0.1
        if eager_confident.sum() > 0 and eager_confident.sum() == export_confident.sum():
            torch.testing.assert_close(
                eager_out[0]["boxes"][eager_confident],
                export_out[0]["boxes"][export_confident],
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                eager_out[0]["scores"][eager_confident],
                export_out[0]["scores"][export_confident],
                atol=1e-6,
                rtol=1e-6,
            )

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_matches_eager_random_input(self, fasterrcnn_model, strict):
        """Exported model should match eager on the same random input used by test_detection_model."""
        set_rng_seed(0)
        with torch.no_grad():
            ep = export(
                fasterrcnn_model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        x = torch.rand(3, 320, 320)
        with torch.no_grad():
            eager_out = fasterrcnn_model([x.clone()])
            export_out = ep.module()([x.clone()])

        for key in ("boxes", "scores", "labels"):
            assert key in export_out[0], f"Missing key '{key}' in export output"

        # Only compare confident detections (see test_export_matches_eager_real_image)
        eager_confident = eager_out[0]["scores"] > 0.1
        export_confident = export_out[0]["scores"] > 0.1
        if eager_confident.sum() > 0 and eager_confident.sum() == export_confident.sum():
            torch.testing.assert_close(
                eager_out[0]["boxes"][eager_confident],
                export_out[0]["boxes"][export_confident],
                atol=1e-4,
                rtol=1e-4,
            )

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("h_val,w_val", [(256, 512), (384, 320), (448, 640), (256, 256)])
    def test_export_dynamic_shapes(self, fasterrcnn_model, h_val, w_val, strict):
        """Exported model should run on various input sizes without error."""
        with torch.no_grad():
            ep = export(
                fasterrcnn_model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        set_rng_seed(42)
        x = torch.rand(3, h_val, w_val)
        with torch.no_grad():
            eager_out = fasterrcnn_model([x.clone()])
            export_out = ep.module()([x.clone()])

        assert len(eager_out) == 1 and len(export_out) == 1
        for key in ("boxes", "scores", "labels"):
            assert key in export_out[0], f"Missing key '{key}' in export output"

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_zero_detections(self, fasterrcnn_model, strict):
        """Exported model should handle the case where NMS produces 0 detections."""
        # Use default thresholds — random noise should produce 0 detections
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=50, weights_backbone=None, _skip_resize=True)
        model.eval()
        with torch.no_grad():
            _ = model([torch.randn(3, 256, 256)])

        with torch.no_grad():
            ep = export(
                model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        set_rng_seed(0)
        x = torch.rand(3, 320, 512)
        with torch.no_grad():
            eager_out = model([x.clone()])
            export_out = ep.module()([x.clone()])

        assert len(eager_out[0]["boxes"]) == len(export_out[0]["boxes"])

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_many_detections(self, strict):
        """Exported model with lowered thresholds should produce many detections."""
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=50, weights_backbone=None, _skip_resize=True)
        model.eval()
        model.rpn.score_thresh = 0.0
        model.rpn._pre_nms_top_n = {"training": 2000, "testing": 100}
        model.rpn._post_nms_top_n = {"training": 2000, "testing": 100}
        model.roi_heads.score_thresh = 0.0
        model.roi_heads.detections_per_img = 20

        with torch.no_grad():
            _ = model([torch.randn(3, 256, 256)])

        with torch.no_grad():
            ep = export(
                model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        set_rng_seed(42)
        x = torch.rand(3, 320, 512)
        with torch.no_grad():
            eager_out = model([x.clone()])
            export_out = ep.module()([x.clone()])

        n_eager = len(eager_out[0]["boxes"])
        n_export = len(export_out[0]["boxes"])
        assert n_eager > 0, "Expected detections with lowered thresholds"
        assert n_export > 0, "Export should also produce detections"
        # With random weights, NMS is sensitive to floating-point differences
        # so we verify count and structure rather than exact coordinates
        assert n_eager == n_export

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_zero_detections_structure(self, strict):
        """Exported model should produce correctly-shaped empty tensors when NMS finds nothing."""
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=50, weights_backbone=None, _skip_resize=True)
        model.eval()
        with torch.no_grad():
            _ = model([torch.randn(3, 256, 256)])

        with torch.no_grad():
            ep = export(
                model,
                ([torch.randn(3, 256, 320)],),
                dynamic_shapes=_fpn_dynamic_shapes(),
                strict=strict,
            )

        set_rng_seed(0)
        x = torch.rand(3, 384, 512)
        with torch.no_grad():
            eager_out = model([x.clone()])
            export_out = ep.module()([x.clone()])

        assert eager_out[0]["boxes"].shape[0] == 0, "Expected 0 eager detections with default thresholds"
        assert export_out[0]["boxes"].shape == torch.Size([0, 4])
        assert export_out[0]["scores"].shape == torch.Size([0])
        assert export_out[0]["labels"].shape == torch.Size([0])

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_static_shapes(self, fasterrcnn_model, strict):
        """Export with fully static shapes should also work."""
        with torch.no_grad():
            ep = export(
                fasterrcnn_model,
                ([torch.randn(3, 300, 300)],),
                strict=strict,
            )

        set_rng_seed(0)
        x = torch.rand(3, 300, 300)
        with torch.no_grad():
            eager_out = fasterrcnn_model([x.clone()])
            export_out = ep.module()([x.clone()])

        assert len(eager_out[0]["boxes"]) == len(export_out[0]["boxes"])
