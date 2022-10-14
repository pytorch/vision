import torch
import torch.nn as nn

from torchvision.prototype import features


class WrapIntoFeatures(torch.nn.Module):
    def forward(self, sample):
        video, target, id = sample
        return features.Video(video), features.Label(target), features._Feature(id)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)
