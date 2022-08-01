import unittest

import pytest
import torch

from torchvision.models.maxvit import SwapAxes, WindowDepartition, WindowPartition


class MaxvitTester(unittest.TestCase):
    def test_maxvit_window_partition(self):
        input_shape = (1, 3, 224, 224)
        partition_size = 7

        x = torch.randn(input_shape)

        partition = WindowPartition(partition_size=7)
        departition = WindowDepartition(partition_size=partition_size, n_partitions=(input_shape[3] // partition_size))

        assert torch.allclose(x, departition(partition(x)))

    def test_maxvit_grid_partition(self):
        input_shape = (1, 3, 224, 224)
        partition_size = 7

        x = torch.randn(input_shape)
        partition = torch.nn.Sequential(
            WindowPartition(partition_size=(input_shape[3] // partition_size)),
            SwapAxes(-2, -3),
        )
        departition = torch.nn.Sequential(
            SwapAxes(-2, -3),
            WindowDepartition(partition_size=(input_shape[3] // partition_size), n_partitions=partition_size),
        )

        assert torch.allclose(x, departition(partition(x)))


if __name__ == "__main__":
    pytest.main([__file__])
