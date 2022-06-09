import torch
from torchvision.ops import PSRoIPool


def test_m1_operator():
    pool_size = 5
    n_channels = 2 * (pool_size ** 2)
    x = torch.rand(2, n_channels, 10, 10, dtype=torch.float64, device='cpu')
    rois = torch.tensor(
        [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],
        dtype=torch.float64,
        device='cpu',
    )

    m = PSRoIPool(pool_size, 1)
    y = m(x, rois)
    assert y.shape == torch.Size([4, 2, 5, 5])
