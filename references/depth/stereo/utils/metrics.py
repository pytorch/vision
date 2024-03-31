from typing import Dict, List, Optional, Tuple

from torch import Tensor

AVAILABLE_METRICS = ["mae", "rmse", "epe", "bad1", "bad2", "epe", "1px", "3px", "5px", "fl-all", "relepe"]


def compute_metrics(
    flow_pred: Tensor, flow_gt: Tensor, valid_flow_mask: Optional[Tensor], metrics: List[str]
) -> Tuple[Dict[str, float], int]:
    for m in metrics:
        if m not in AVAILABLE_METRICS:
            raise ValueError(f"Invalid metric: {m}. Valid metrics are: {AVAILABLE_METRICS}")

    metrics_dict = {}

    pixels_diffs = (flow_pred - flow_gt).abs()
    # there is no Y flow in Stereo Matching, therefore flow.abs() = flow.pow(2).sum(dim=1).sqrt()
    flow_norm = flow_gt.abs()

    if valid_flow_mask is not None:
        valid_flow_mask = valid_flow_mask.unsqueeze(1)
        pixels_diffs = pixels_diffs[valid_flow_mask]
        flow_norm = flow_norm[valid_flow_mask]

    num_pixels = pixels_diffs.numel()
    if "bad1" in metrics:
        metrics_dict["bad1"] = (pixels_diffs > 1).float().mean().item()
    if "bad2" in metrics:
        metrics_dict["bad2"] = (pixels_diffs > 2).float().mean().item()

    if "mae" in metrics:
        metrics_dict["mae"] = pixels_diffs.mean().item()
    if "rmse" in metrics:
        metrics_dict["rmse"] = pixels_diffs.pow(2).mean().sqrt().item()
    if "epe" in metrics:
        metrics_dict["epe"] = pixels_diffs.mean().item()
    if "1px" in metrics:
        metrics_dict["1px"] = (pixels_diffs < 1).float().mean().item()
    if "3px" in metrics:
        metrics_dict["3px"] = (pixels_diffs < 3).float().mean().item()
    if "5px" in metrics:
        metrics_dict["5px"] = (pixels_diffs < 5).float().mean().item()
    if "fl-all" in metrics:
        metrics_dict["fl-all"] = ((pixels_diffs < 3) & ((pixels_diffs / flow_norm) < 0.05)).float().mean().item() * 100
    if "relepe" in metrics:
        metrics_dict["relepe"] = (pixels_diffs / flow_norm).mean().item()

    return metrics_dict, num_pixels
