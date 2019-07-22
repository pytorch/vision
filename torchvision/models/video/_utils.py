import torch.nn as nn


__all__ = ["Conv3DSimple", "Conv3DNoTemporal"]


class Conv3DSimple(nn.Module):
    def __init__(self):
        super(Conv3DSimple, self).__init__()
    
    def get_conv(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

class Conv2Plus1D(nn.Module):
    """Generate separated convolution as in https://arxiv.org/abs/1711.11248

    Args:
        in_planes (int): Input channels
        out_planes (int): Output channels
        midplanes (int): Step-up channels to match the param number
        stride (int, optional): Convolution striding. Defaults to 1.
        padding (int, optional): Pad the conv layer. Defaults to 1.

    """
    def __init__(self):
        super(Conv2Plus1D, self).__init__()

    def get_conv(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
   
        return nn.Sequential(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False)
            )

    def forward(self, x):
        return self.conv1(x)


class Conv3DNoTemporal(nn.Module):
    def __init__(self):
        super(Conv3DNoTemporal, self).__init__()

    def get_conv(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    def forward(self, x):
        return self.conv1(x)
