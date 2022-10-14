import torch
import torch.nn as nn

from torchvision.prototype import features


class WrapIntoFeatures(torch.nn.Module):
    def forward(self, sample):
        video, target, id = sample
        video = video.transpose(-4, -3)  # convert back to (B, C, H, W)
        return features.Video(video), features.Label(target), features._Feature(id)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, *inputs):
        inputs[0].transpose_(-4, -3)
        return inputs
