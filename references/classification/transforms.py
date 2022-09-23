from torch import nn
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F


class WrapIntoFeatures(nn.Module):
    def forward(self, sample):
        image, target = sample
        return F.to_image_tensor(image), features.Label(target)
