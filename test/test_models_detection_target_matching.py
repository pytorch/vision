import pytest
import torch
from torchvision.models.detection.anchor_utils import grid_centers
from torchvision.models.detection.target_matching import aligned_iou, iou_below, is_inside_box, _sim_ota_match


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

    is_inside[0]:
        [[F, F, F, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]]

    is_inside[1]:
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
    is_inside = is_inside_box(centers, boxes).view(2, 5, 10)
    assert torch.count_nonzero(is_inside) == 6
    assert torch.all(is_inside[0, 1:3, 1:3])
    assert torch.all(is_inside[1, 4, 7:9])


def test_sim_ota_match():
    # IoUs will determined that 2 and 1 predictions will be selected for the first and the second target.
    ious = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.2, 0.3, 0.4, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. Since the first prediction was selected for both targets, it
    # will be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.5, 0.4, 0.3], [0.1, 0.2, 0.5, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]
    assert len(matched_targets) == 2  # Two predictions were matched.
    assert matched_targets[0] == 1  # Which target was matched to the first prediction.
    assert matched_targets[1] == 0  # Which target was matched to the last prediction.
