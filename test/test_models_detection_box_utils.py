import pytest
import torch
from torchvision.models.detection.anchor_utils import grid_centers
from torchvision.models.detection.box_utils import aligned_iou, box_size_ratio, iou_below, is_inside_box


@pytest.mark.parametrize(
    "dims1, dims2, expected_ious",
    [
        (
            torch.tensor([[1.0, 1.0], [10.0, 1.0], [100.0, 10.0]]),
            torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            torch.tensor([[1.0 / 10.0, 1.0 / 40.0], [1.0 / 19.0, 2.0 / 48.0], [10.0 / 1000.0, 20.0 / 1020.0]]),
        )
    ],
)
def test_aligned_iou(dims1, dims2, expected_ious):
    torch.testing.assert_close(aligned_iou(dims1, dims2), expected_ious)


def test_iou_below():
    tl = torch.rand((10, 10, 3, 2)) * 100
    br = tl + 10
    pred_boxes = torch.cat((tl, br), -1)
    target_boxes = torch.stack((pred_boxes[1, 1, 0], pred_boxes[3, 5, 1]))
    result = iou_below(pred_boxes, target_boxes, 0.9)
    assert result.shape == (10, 10, 3)
    assert not result[1, 1, 0]
    assert not result[3, 5, 1]


def test_is_inside_box():
    """
    centers:
        [[1,1; 3,1; 5,1; 7,1; 9,1; 11,1; 13,1; 15,1; 17,1; 19,1]
         [1,3; 3,3; 5,3; 7,3; 9,3; 11,3; 13,3; 15,3; 17,3; 19,3]
         [1,5; 3,5; 5,5; 7,5; 9,5; 11,5; 13,5; 15,5; 17,5; 19,5]
         [1,7; 3,7; 5,7; 7,7; 9,7; 11,7; 13,7; 15,7; 17,7; 19,7]
         [1,9; 3,9; 5,9; 7,9; 9,9; 11,9; 13,9; 15,9; 17,9; 19,9]]

    is_inside[..., 0]:
        [[F, F, F, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]]

    is_inside[..., 1]:
        [[F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, T, T, F]]
    """
    size = torch.tensor([10, 5])
    centers = grid_centers(size) * 2.0
    centers = centers.view(-1, 2)
    boxes = torch.tensor([[2, 2, 6, 6], [14, 8, 18, 10]])
    is_inside = is_inside_box(centers, boxes).view(5, 10, 2)
    assert torch.count_nonzero(is_inside) == 6
    assert torch.all(is_inside[1:3, 1:3, 0])
    assert torch.all(is_inside[4, 7:9, 1])


def test_box_size_ratio():
    wh1 = torch.tensor([[24, 11], [12, 25], [26, 27], [15, 17]])
    wh2 = torch.tensor([[10, 30], [15, 9]])
    result = box_size_ratio(wh1, wh2)
    assert result.shape == (4, 2)
    assert result[0, 0] == 30 / 11
    assert result[0, 1] == 24 / 15
    assert result[1, 0] == 12 / 10
    assert result[1, 1] == 25 / 9
    assert result[2, 0] == 26 / 10
    assert result[2, 1] == 27 / 9
    assert result[3, 0] == 30 / 17
    assert result[3, 1] == 17 / 9
