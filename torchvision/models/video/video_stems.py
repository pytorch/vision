import torch.nn as nn


def get_default_stem(use_pool1=False):
    """The default conv-batchnorm-relu(-maxpool) stem

    Args:
        use_pool1 (bool, optional): Should the stem include the default maxpool? Defaults to False.

    Returns:
        nn.Sequential: Conv1 stem of resnet based models.
    """

    m = [
        nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                  padding=(1, 3, 3), bias=False),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True)]
    if use_pool1:
        m.append(nn. MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1))
    return nn.Sequential(*m)


def get_r2plus1d_stem(use_pool1=False):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution

    Args:
        use_pool1 (bool, optional): Should the stem contain pool1 layer. Defaults to False.

    Returns:
        nn.Sequential: the stem of the conv-separated network.
    """

    m = [
        nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                  stride=(1, 2, 2), padding=(0, 3, 3),
                  bias=False),
        nn.BatchNorm3d(45),
        nn.ReLU(inplace=True),
        nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                  stride=(1, 1, 1), padding=(1, 0, 0),
                  bias=False),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True)]

    if use_pool1:
        m.append(nn. MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1))
    return nn.Sequential(*m)
