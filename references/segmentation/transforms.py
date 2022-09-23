import torch

from torchvision.prototype import features, transforms
from torchvision.prototype.transforms import functional as F


class WrapIntoFeatures(transforms.Transform):
    def forward(self, sample):
        image, mask = sample
        return F.to_image_tensor(image), features.Mask(F.pil_to_tensor(mask).squeeze(0), dtype=torch.int64)


class PadIfSmaller(transforms.Transform):
    def __init__(self, size, fill=0):
        super().__init__()
        self.size = size
        self.fill = transforms._geometry._setup_fill_arg(fill)

    def _get_params(self, sample):
        _, height, width = transforms._utils.query_chw(sample)
        padding = [0, 0, max(self.size - width, 0), max(self.size - height, 0)]
        needs_padding = any(padding)
        return dict(padding=padding, needs_padding=needs_padding)

    def _transform(self, inpt, params):
        if not params["needs_padding"]:
            return inpt

        fill = self.fill[type(inpt)]
        fill = F._geometry._convert_fill_arg(fill)

        return F.pad(inpt, padding=params["padding"], fill=fill)
