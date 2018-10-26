import torch
from torchvision.datasets import MNIST, FashionMNIST
import unittest
import tempfile
import shutil
import os


class Tester(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fashion_mnist_doesnt_load_mnist(self):
        MNIST(root=self.test_dir, download=True)
        FashionMNIST(root=self.test_dir, download=True)


if __name__ == '__main__':
    unittest.main()
