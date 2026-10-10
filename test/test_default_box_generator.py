"""
Tests for DefaultBoxGenerator — specifically the CUDA device-mismatch fix.

Covers:
  - CPU output shape, device, and value sanity
  - clip=True clamps boxes to [0, 1] normalized coords (via _wh_pairs)
  - clip=False allows boxes outside [0, 1]
  - Batch size > 1
  - CUDA device consistency (skipped when CUDA unavailable)
  - TorchScript compatibility on CPU
  - TorchScript compatibility on CUDA (skipped when CUDA unavailable)

Run with:
    pytest test_default_box_generator.py -v
or for CUDA tests specifically:
    pytest test_default_box_generator.py -v -k cuda
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList

# ---------------------------------------------------------------------------
# Constants — SSD-300 configuration (well-known: produces exactly 8 732 anchors)
# ---------------------------------------------------------------------------

ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
FEATURE_MAP_SIZES = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
IMAGE_SIZE = (300, 300)
EXPECTED_BOXES = 8732  # sum of (h*w * num_anchors_per_cell) across all feature maps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(device: torch.device, batch_size: int = 1):
    """Return (ImageList, feature_maps) on the requested device."""
    image_tensors = torch.zeros(batch_size, 3, *IMAGE_SIZE, device=device)
    image_sizes = [IMAGE_SIZE] * batch_size
    image_list = ImageList(image_tensors, image_sizes)
    feature_maps = [
        torch.zeros(batch_size, 1, h, w, device=device)
        for h, w in FEATURE_MAP_SIZES
    ]
    return image_list, feature_maps


def make_generator(clip: bool = True) -> DefaultBoxGenerator:
    return DefaultBoxGenerator(aspect_ratios=ASPECT_RATIOS, clip=clip)


# ---------------------------------------------------------------------------
# CPU tests (always run)
# ---------------------------------------------------------------------------

class TestDefaultBoxGeneratorCPU:
    """CPU-only tests — no CUDA required."""

    def test_output_length_matches_batch(self):
        """One anchor list per image in the batch."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"), batch_size=2)
        out = gen(image_list, feature_maps)
        assert len(out) == 2

    def test_output_shape_single_image(self):
        """Each anchor list has shape (EXPECTED_BOXES, 4)."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out = gen(image_list, feature_maps)
        assert out[0].shape == (EXPECTED_BOXES, 4), (
            f"Expected ({EXPECTED_BOXES}, 4), got {out[0].shape}"
        )

    def test_output_shape_batch(self):
        """Shape holds for every image in a batch."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"), batch_size=3)
        out = gen(image_list, feature_maps)
        for i, anchors in enumerate(out):
            assert anchors.shape == (EXPECTED_BOXES, 4), (
                f"Image {i}: expected ({EXPECTED_BOXES}, 4), got {anchors.shape}"
            )

    def test_output_device_is_cpu(self):
        """Anchors must be on CPU when inputs are on CPU."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out = gen(image_list, feature_maps)
        assert out[0].device.type == "cpu"

    def test_no_nans(self):
        """Anchor coordinates must be finite (no NaN)."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out = gen(image_list, feature_maps)
        assert not torch.any(torch.isnan(out[0])), "NaN values found in anchors"

    def test_clip_true_clamps_wh_pairs(self):
        """
        With clip=True the internal _wh_pairs are clamped to [0, 1], so
        the (w, h) components of every anchor should be in [0, 1].
        The (cx, cy) components can legitimately be outside that range
        for border anchors, so we only check wh.
        """
        gen = make_generator(clip=True)
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out = gen(image_list, feature_maps)
        # out is in (x1, y1, x2, y2) pixel space — convert width/height
        boxes = out[0]  # (N, 4): x1, y1, x2, y2
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        # pixel widths/heights must be ≤ image dimension (clip keeps wh ≤ 1 in normalized)
        assert (widths <= IMAGE_SIZE[1] + 1e-4).all(), "Width exceeds image width"
        assert (heights <= IMAGE_SIZE[0] + 1e-4).all(), "Height exceeds image height"

    def test_clip_false_allows_large_anchors(self):
        """With clip=False, some anchors can be larger than the image."""
        gen_clipped = make_generator(clip=True)
        gen_free = make_generator(clip=False)
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out_clipped = gen_clipped(image_list, feature_maps)
        out_free = gen_free(image_list, feature_maps)
        # Both should still have the right shape
        assert out_clipped[0].shape == (EXPECTED_BOXES, 4)
        assert out_free[0].shape == (EXPECTED_BOXES, 4)
        # clip=False boxes may be larger — widths can exceed image size
        boxes_free = out_free[0]
        widths_free = boxes_free[:, 2] - boxes_free[:, 0]
        assert (widths_free > IMAGE_SIZE[1]).any(), (
            "Expected some anchors wider than image when clip=False"
        )

    def test_output_dtype_float32(self):
        """Default dtype should be float32."""
        gen = make_generator()
        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out = gen(image_list, feature_maps)
        assert out[0].dtype == torch.float32

    def test_torchscript_cpu(self):
        """DefaultBoxGenerator must be TorchScript-traceable on CPU."""
        gen = make_generator()
        gen.eval()
        scripted = torch.jit.script(gen)

        image_list, feature_maps = make_inputs(torch.device("cpu"))
        out_eager = gen(image_list, feature_maps)
        out_scripted = scripted(image_list, feature_maps)

        assert len(out_scripted) == len(out_eager)
        torch.testing.assert_close(out_scripted[0], out_eager[0])


# ---------------------------------------------------------------------------
# CUDA tests (skipped when CUDA unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDefaultBoxGeneratorCUDA:
    """
    CUDA tests — verify fix for issue #9414:
    DefaultBoxGenerator._grid_default_boxes previously built shifts on CPU
    and left self._wh_pairs on CPU, causing torch.cat to raise a device
    mismatch error when the model was moved to GPU.
    """

    def test_output_device_is_cuda(self):
        """
        Core regression test for #9414.
        Anchors must be on the same CUDA device as the input feature maps.
        Before the fix this raised: RuntimeError: Expected all tensors to be
        on the same device, but found at least two devices, cpu and cuda:0!
        """
        gen = make_generator()
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)
        out = gen(image_list, feature_maps)  # must not raise
        assert out[0].device.type == "cuda", (
            f"Anchors on wrong device: {out[0].device} (expected cuda)"
        )

    def test_cuda_output_shape(self):
        """Shape is correct on CUDA."""
        gen = make_generator()
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)
        out = gen(image_list, feature_maps)
        assert out[0].shape == (EXPECTED_BOXES, 4)

    def test_cuda_no_nans(self):
        """No NaN values on CUDA."""
        gen = make_generator()
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)
        out = gen(image_list, feature_maps)
        assert not torch.any(torch.isnan(out[0]))

    def test_cuda_cpu_values_match(self):
        """
        Anchor coordinates produced on CUDA must match those on CPU
        (up to floating-point tolerance), confirming no device-dependent
        numeric divergence was introduced by the fix.
        """
        gen = make_generator()
        device = torch.device("cuda:0")

        image_list_cpu, fmaps_cpu = make_inputs(torch.device("cpu"))
        image_list_gpu, fmaps_gpu = make_inputs(device)

        out_cpu = gen(image_list_cpu, fmaps_cpu)
        out_gpu = gen(image_list_gpu, fmaps_gpu)

        torch.testing.assert_close(
            out_gpu[0].cpu(), out_cpu[0],
            atol=1e-5, rtol=1e-5,
            msg="CPU and CUDA anchors differ beyond tolerance",
        )

    def test_cuda_batch(self):
        """Batch of 2 works correctly on CUDA."""
        gen = make_generator()
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device, batch_size=2)
        out = gen(image_list, feature_maps)
        assert len(out) == 2
        for anchors in out:
            assert anchors.shape == (EXPECTED_BOXES, 4)
            assert anchors.device.type == "cuda"

    def test_cuda_clip_false(self):
        """clip=False works on CUDA without raising."""
        gen = make_generator(clip=False)
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)
        out = gen(image_list, feature_maps)  # must not raise
        assert out[0].shape == (EXPECTED_BOXES, 4)
        assert out[0].device.type == "cuda"

    def test_torchscript_cuda(self):
        """TorchScript tracing works on CUDA and results match eager mode."""
        gen = make_generator()
        gen.eval()
        scripted = torch.jit.script(gen)

        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)

        out_eager = gen(image_list, feature_maps)
        out_scripted = scripted(image_list, feature_maps)

        assert len(out_scripted) == len(out_eager)
        torch.testing.assert_close(out_scripted[0], out_eager[0])

    def test_wh_pairs_moved_to_cuda(self):
        """
        Internal _wh_pairs (created on CPU in __init__) must be transparently
        moved to CUDA during forward — confirmed by the anchors being on CUDA.
        This is the exact mechanism broken by #9414.
        """
        gen = make_generator()
        # _wh_pairs starts on CPU (by design in __init__)
        assert gen._wh_pairs.device.type == "cpu", (
            "_wh_pairs should be on CPU after __init__"
        )
        device = torch.device("cuda:0")
        image_list, feature_maps = make_inputs(device)
        out = gen(image_list, feature_maps)
        # After forward on CUDA, output must be on CUDA
        assert out[0].device.type == "cuda"
        # _wh_pairs itself should still be on CPU (we .to() inside forward, not move the buffer)
        assert gen._wh_pairs.device.type == "cpu", (
            "_wh_pairs should remain on CPU after forward (we .to() inside the call, not mutate)"
        )
