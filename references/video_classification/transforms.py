import torch
import torch.nn as nn

from torchvision.prototype import features


class WrapIntoFeatures(torch.nn.Module):
    def forward(self, sample):
        video_cthw, target, id = sample
        video_tchw = video_cthw.transpose(-4, -3)
        return features.Video(video_tchw), features.Label(target), features._Feature(id)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)
