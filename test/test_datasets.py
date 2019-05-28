import PIL
import shutil
import tempfile
import unittest

import torchvision


class Tester(unittest.TestCase):

    def test_mnist(self):
        tmp_dir = tempfile.mkdtemp()
        dataset = torchvision.datasets.MNIST(tmp_dir, download=True)
        self.assertEqual(len(dataset), 60000)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        shutil.rmtree(tmp_dir)

    def test_emnist(self):
        tmp_dir = tempfile.mkdtemp()
        dataset = torchvision.datasets.EMNIST(tmp_dir, split='byclass', download=True)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        shutil.rmtree(tmp_dir)

    def test_kmnist(self):
        tmp_dir = tempfile.mkdtemp()
        dataset = torchvision.datasets.KMNIST(tmp_dir, download=True)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        shutil.rmtree(tmp_dir)

    def test_fashionmnist(self):
        tmp_dir = tempfile.mkdtemp()
        dataset = torchvision.datasets.FashionMNIST(tmp_dir, download=True)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()
