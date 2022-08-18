from torch import nn
from torchvision.prototype import features


class WrapIntoFeatures(nn.Module):
    def forward(self, sample):
        input, target = sample
        return features.Image(input), features.Label(target)
