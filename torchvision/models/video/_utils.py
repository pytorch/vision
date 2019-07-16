import torch.nn as nn

def video_3d_conv(
    in_planes,
    out_planes,
    midplanes=None,
    stride=1,
    padding=1):
    """An everyday 3D convolution generator
    
    Args:
        in_planes (int): Num channels in
        out_planes (int): Num channels out
        midplanes (int, optional): In the case of R(2+1)D, we adjust the channels between the convolutions. Defaults to None.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Convolution padding. Defaults to 1.
    
    Returns:
        nn.Conv3d: Conv3D object
    """

    return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)


def video_2plus1d_conv(
    in_planes,
    out_planes,
    midplanes,
    stride=1,
    padding=1):
    """Generate separated convolution as in https://arxiv.org/abs/1711.11248

    Args:
        in_planes (int): Input channels
        out_planes (int): Output channels
        midplanes (int): Step-up channels to match the param number
        stride (int, optional): Convolution striding. Defaults to 1.
        padding (int, optional): Pad the conv layer. Defaults to 1.

    Returns:
        nn.Sequential: Separated 2+1 convolution layer to be a part of the video trunk block
    """

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

def video_2d_conv(
    in_planes,
    out_planes,
    midplanes=None,
    stride=1,
    padding=1):
    """2D convolution for video can be implemented with 3d conv ops;
    we do so to avoid tensor transforms in the model. 
    
    Args:
        in_planes (int): Num channels in
        out_planes (int): Num channels out
        midplanes (int, optional): In the case of R(2+1)D, we adjust the channels between the convolutions. Defaults to None.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Convolution padding. Defaults to 1.
    
    Returns:
        nn.Conv3d: Conv3D object
    """

    return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)
