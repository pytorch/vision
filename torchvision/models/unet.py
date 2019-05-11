import torch
import torch.nn as nn


__all__ = ['UNet', 'unet8', 'unet13', 'unet18', 'unet23', 'unet28', 'unet33']


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU()
    )


def center_crop(img, output_size):
    _, _, h, w = img.size()
    _, _, th, tw = output_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return img[:, :, i:i + th, j:j + tw]


class Contract(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, p=0.5):
        super(Contract, self).__init__()
        assert in_channels < out_channels

        self.pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_channels, out_channels)
        self.drop = None

        if dropout:
            self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        if self.drop is not None:
            x = self.drop(x)

        return x


class Expand(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Expand, self).__init__()
        assert in_channels > out_channels

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.relu = nn.ReLU()
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x, out):
        x = self.upconv(x)
        x = self.relu(x)

        out = center_crop(out, x.size())
        x = torch.cat([out, x], 1)

        x = self.conv(x)

        return x


class UNet(nn.Module):
    """`U-Net <https://arxiv.org/pdf/1505.04597.pdf>`_ architecture.

    Args:
        in_channels (int, optional): number of channels in input image
        out_channels (int, optional): number of channels in output segmentation
        start_channels (int, optional): power of 2 channels to start with
        depth (int, optional): number of contractions/expansions
        p (float, optional): dropout probability
    """

    def __init__(self, in_channels=1, out_channels=2, start_channels=6,
                 depth=4, p=0.5):
        super(UNet, self).__init__()

        self.depth = depth

        # Contraction
        self.conv1 = double_conv(in_channels, 2 ** start_channels)
        self.contractions = nn.ModuleList([
            Contract(2 ** d, 2 ** (d + 1), dropout=d - depth > 3, p=p)
            for d in range(start_channels, start_channels + depth)
        ])

        # Expansion
        self.expansions = nn.ModuleList([
            Expand(2 ** d, 2 ** (d - 1)) for d in range(
                start_channels + depth, start_channels, -1)
        ])
        self.conv2 = nn.Conv2d(2 ** start_channels, out_channels, 1)
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Contraction
        out = [self.conv1(x)]
        for f in self.contractions:
            out.append(f(out[-1]))

        # Expansion
        i = -2
        x = out[-1]
        for f in self.expansions:
            x = f(x, out[i])
            i -= 1

        x = self.conv2(x)
        x = self.softmax(x)

        return x


def unet8(**kwargs):
    """Constructs a U-Net 8 model."""
    return UNet(depth=1, **kwargs)


def unet13(**kwargs):
    """Constructs a U-Net 13 model."""
    return UNet(depth=2, **kwargs)


def unet18(**kwargs):
    """Constructs a U-Net 18 model."""
    return UNet(depth=3, **kwargs)


def unet23(**kwargs):
    """Constructs a U-Net 23 model."""
    return UNet(depth=4, **kwargs)


def unet28(**kwargs):
    """Constructs a U-Net 28 model."""
    return UNet(depth=5, **kwargs)


def unet33(**kwargs):
    """Constructs a U-Net 33 model."""
    return UNet(depth=6, **kwargs)
