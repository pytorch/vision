import torch

from torchvision.models.detection.target_matching import _sim_ota_match


def test_sim_ota_match():
    # For each of the two targets, k will be the sum of the IoUs. 2 and 1 predictions will be selected for the first and
    # the second target respectively.
    ious = torch.tensor([[0.1, 0.2], [0.1, 0.3], [0.9, 0.4], [0.9, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. The first prediction was selected for two targets, but it will
    # be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.1], [0.5, 0.2], [0.4, 0.5], [0.3, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)

    # The first and the last prediction were matched.
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]

    # The first prediction was matched to the target 1 and the last prediction was matched to target 0.
    assert len(matched_targets) == 2
    assert matched_targets[0] == 1
    assert matched_targets[1] == 0
