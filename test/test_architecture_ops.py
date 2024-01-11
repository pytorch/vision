import unittest

import pytest
import torch

from torchvision.models.maxvit import SwapAxes, WindowDepartition, WindowPartition


class MaxvitTester(unittest.TestCase):
    def test_maxvit_window_partition(self):
        input_shape = (1, 3, 224, 224)
        partition_size = 7
        n_partitions = input_shape[3] // partition_size

        x = torch.randn(input_shape)

        partition = WindowPartition()
        departition = WindowDepartition()

        x_hat = partition(x, partition_size)
        x_hat = departition(x_hat, partition_size, n_partitions, n_partitions)

        torch.testing.assert_close(x, x_hat)

    def test_maxvit_grid_partition(self):
        input_shape = (1, 3, 224, 224)
        partition_size = 7
        n_partitions = input_shape[3] // partition_size

        x = torch.randn(input_shape)
        pre_swap = SwapAxes(-2, -3)
        post_swap = SwapAxes(-2, -3)

        partition = WindowPartition()
        departition = WindowDepartition()

        x_hat = partition(x, n_partitions)
        x_hat = pre_swap(x_hat)
        x_hat = post_swap(x_hat)
        x_hat = departition(x_hat, n_partitions, partition_size, partition_size)

        torch.testing.assert_close(x, x_hat)


if __name__ == "__main__":
    pytest.main([__file__])
