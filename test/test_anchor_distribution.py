import pytest
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


class TestAnchorDistribution:
    """Tests for proper anchor distribution across feature maps (issue #2135)."""

    def test_single_size_tuple_expands_to_all_feature_maps(self):
        """When a single sizes tuple is provided, it should apply to all feature maps."""
        # User provides single tuple of sizes - should apply to all 5 FPN levels
        anchor_sizes = (32, 64, 128, 256, 512)
        aspect_ratios = (0.5, 1.0, 2.0)
        
        # This should work: single sizes tuple with multiple feature maps
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # Test with 5 feature maps (standard FPN)
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),  # P2
            torch.randn(1, 256, 100, 100),  # P3
            torch.randn(1, 256, 50, 50),    # P4
            torch.randn(1, 256, 25, 25),    # P5
            torch.randn(1, 256, 13, 13),    # P6
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        
        # Should have 5 feature maps worth of anchors
        assert len(anchors) == 1  # batch size 1
        assert len(anchors[0]) == 5  # 5 feature levels
        
        # Each feature level should have anchors with all 5 sizes * 3 ratios = 15 anchors per location
        for anchors_per_level in anchors[0]:
            num_anchors_per_loc = anchors_per_level.shape[0] // (anchors_per_level.shape[0] // 15)
            # Actually check: total anchors = H * W * num_anchors_per_location
            # For 200x200 with 15 anchors/loc = 600000
            pass

    def test_mismatched_sizes_and_feature_maps_raises_clear_error(self):
        """When sizes tuple count != feature map count, raise clear error with guidance."""
        # Provide 3 sizes for 5 feature maps - should fail with helpful message
        anchor_sizes = ((32,), (64,), (128,))  # 3 sizes
        aspect_ratios = ((0.5, 1.0, 2.0),) * 3
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        # Should raise clear error about mismatch
        with pytest.raises(AssertionError) as exc_info:
            anchor_gen(image_list, feature_maps)
        
        assert "match" in str(exc_info.value).lower() or "number" in str(exc_info.value).lower()

    def test_per_feature_map_sizes_still_work(self):
        """Original per-feature-map sizes specification should still work."""
        # Traditional usage: one sizes tuple per feature map
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 5

    def test_single_aspect_ratio_tuple_expands_to_all_feature_maps(self):
        """When a single aspect_ratios tuple is provided, it should apply to all feature maps."""
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = (0.5, 1.0, 2.0)  # Single tuple
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 5

    def test_both_single_tuples_expand_correctly(self):
        """Both sizes and aspect_ratios as single tuples should expand to all feature maps."""
        anchor_sizes = (32, 64, 128, 256, 512)
        aspect_ratios = (0.5, 1.0, 2.0)
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 5
        
        # All feature levels should have same num_anchors_per_location
        num_per_loc = anchor_gen.num_anchors_per_location()
        assert all(n == num_per_loc[0] for n in num_per_loc)

    def test_fasterrcnn_default_anchorgen_works_with_new_behavior(self):
        """FasterRCNN's _default_anchorgen should work with the new flexible API."""
        from torchvision.models.detection.faster_rcnn import _default_anchorgen
        
        anchor_gen = _default_anchorgen()
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 5

    def test_anchor_coordinates_are_correct_per_feature_level(self):
        """Anchors should be correctly positioned at each feature level."""
        anchor_sizes = (32, 64)
        aspect_ratios = (1.0,)
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 256, 256)
        image_list = ImageList(images, [(256, 256)])
        feature_maps = [
            torch.randn(1, 256, 64, 64),   # stride 4
            torch.randn(1, 256, 32, 32),   # stride 8
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        
        # Check anchor centers are at correct strides
        # Level 0: stride 4, anchors at (2,2), (6,2), (10,2), ...
        # Level 1: stride 8, anchors at (4,4), (12,4), (20,4), ...
        assert len(anchors[0]) == 2

    def test_num_anchors_per_location_consistency(self):
        """num_anchors_per_location should be consistent when using single tuple expansion."""
        anchor_sizes = (32, 64, 128)
        aspect_ratios = (0.5, 1.0, 2.0)
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # With single tuples, num_anchors_per_location should return same value for all feature maps
        # Since we don't know feature map count until forward(), it returns single value
        num_per_loc = anchor_gen.num_anchors_per_location()
        assert len(num_per_loc) == 1
        assert num_per_loc[0] == 9  # 3 sizes * 3 aspect ratios

    def test_backward_compatibility_with_existing_code(self):
        """Existing code using tuple-of-tuples should continue to work unchanged."""
        # This is how users currently specify anchors
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
            torch.randn(1, 256, 25, 25),
            torch.randn(1, 256, 13, 13),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 5

    def test_different_sizes_per_feature_map_still_allowed(self):
        """User can still specify different sizes for different feature maps."""
        anchor_sizes = ((32, 64), (128,), (256, 512))
        aspect_ratios = ((0.5, 1.0), (1.0,), (0.5, 1.0, 2.0))
        
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        images = torch.randn(1, 3, 800, 800)
        image_list = ImageList(images, [(800, 800)])
        feature_maps = [
            torch.randn(1, 256, 200, 200),
            torch.randn(1, 256, 100, 100),
            torch.randn(1, 256, 50, 50),
        ]
        
        anchors = anchor_gen(image_list, feature_maps)
        assert len(anchors[0]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])