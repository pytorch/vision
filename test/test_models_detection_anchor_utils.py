import pytest
import torch
from common_utils import assert_equal
from torchvision.models.detection.anchor_utils import (
    AnchorGenerator,
    DefaultBoxGenerator,
    global_xy,
    grid_centers,
    grid_offsets,
)
from torchvision.models.detection.image_list import ImageList


class Tester:
    def test_incorrect_anchors(self):
        incorrect_sizes = (
            (2, 4, 8),
            (32, 8),
        )
        incorrect_aspects = (0.5, 1.0)
        anc = AnchorGenerator(incorrect_sizes, incorrect_aspects)
        image1 = torch.randn(3, 800, 800)
        image_list = ImageList(image1, [(800, 800)])
        feature_maps = [torch.randn(1, 50)]
        pytest.raises(AssertionError, anc, image_list, feature_maps)

    def _init_test_anchor_generator(self):
        anchor_sizes = ((10,),)
        aspect_ratios = ((1,),)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return anchor_generator

    def _init_test_defaultbox_generator(self):
        aspect_ratios = [[2]]
        dbox_generator = DefaultBoxGenerator(aspect_ratios)

        return dbox_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 15, 15)
        features = self.get_features(images)
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(images, features)

        # Estimate the number of target anchors
        grid_sizes = [f.shape[-2:] for f in features]
        num_anchors_estimated = 0
        for sizes, num_anchors_per_loc in zip(grid_sizes, model.num_anchors_per_location()):
            num_anchors_estimated += sizes[0] * sizes[1] * num_anchors_per_loc

        anchors_output = torch.tensor(
            [
                [-5.0, -5.0, 5.0, 5.0],
                [0.0, -5.0, 10.0, 5.0],
                [5.0, -5.0, 15.0, 5.0],
                [-5.0, 0.0, 5.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 0.0, 15.0, 10.0],
                [-5.0, 5.0, 5.0, 15.0],
                [0.0, 5.0, 10.0, 15.0],
                [5.0, 5.0, 15.0, 15.0],
            ]
        )

        assert num_anchors_estimated == 9
        assert len(anchors) == 2
        assert tuple(anchors[0].shape) == (9, 4)
        assert tuple(anchors[1].shape) == (9, 4)
        assert_equal(anchors[0], anchors_output)
        assert_equal(anchors[1], anchors_output)

    def test_defaultbox_generator(self):
        images = torch.zeros(2, 3, 15, 15)
        features = [torch.zeros(2, 8, 1, 1)]
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_defaultbox_generator()
        model.eval()
        dboxes = model(images, features)

        dboxes_output = torch.tensor(
            [
                [6.3750, 6.3750, 8.6250, 8.6250],
                [4.7443, 4.7443, 10.2557, 10.2557],
                [5.9090, 6.7045, 9.0910, 8.2955],
                [6.7045, 5.9090, 8.2955, 9.0910],
            ]
        )

        assert len(dboxes) == 2
        assert tuple(dboxes[0].shape) == (4, 4)
        assert tuple(dboxes[1].shape) == (4, 4)
        torch.testing.assert_close(dboxes[0], dboxes_output, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(dboxes[1], dboxes_output, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("width,height", [(10, 5)])
def test_grid_offsets(width: int, height: int):
    size = torch.tensor([width, height])
    offsets = grid_offsets(size)
    assert offsets.shape == (height, width, 2)
    assert torch.equal(offsets[0, :, 0], torch.arange(width, dtype=offsets.dtype))
    assert torch.equal(offsets[0, :, 1], torch.zeros(width, dtype=offsets.dtype))
    assert torch.equal(offsets[:, 0, 0], torch.zeros(height, dtype=offsets.dtype))
    assert torch.equal(offsets[:, 0, 1], torch.arange(height, dtype=offsets.dtype))


@pytest.mark.parametrize("width,height", [(10, 5)])
def test_grid_centers(width: int, height: int):
    size = torch.tensor([width, height])
    centers = grid_centers(size)
    assert centers.shape == (height, width, 2)
    assert torch.equal(centers[0, :, 0], 0.5 + torch.arange(width, dtype=torch.float))
    assert torch.equal(centers[0, :, 1], 0.5 * torch.ones(width))
    assert torch.equal(centers[:, 0, 0], 0.5 * torch.ones(height))
    assert torch.equal(centers[:, 0, 1], 0.5 + torch.arange(height, dtype=torch.float))


def test_global_xy():
    xy = torch.ones((2, 4, 4, 3, 2)) * 0.5  # 4x4 grid of coordinates to the center of the cell.
    image_size = torch.tensor([400, 200])
    xy = global_xy(xy, image_size)
    assert xy.shape == (2, 4, 4, 3, 2)
    assert torch.all(xy[:, :, 0, :, 0] == 50)
    assert torch.all(xy[:, 0, :, :, 1] == 25)
    assert torch.all(xy[:, :, 1, :, 0] == 150)
    assert torch.all(xy[:, 1, :, :, 1] == 75)
    assert torch.all(xy[:, :, 2, :, 0] == 250)
    assert torch.all(xy[:, 2, :, :, 1] == 125)
    assert torch.all(xy[:, :, 3, :, 0] == 350)
    assert torch.all(xy[:, 3, :, :, 1] == 175)
