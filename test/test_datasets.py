import PIL
import os
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

    def test_celeba(self):
        temp_dir = tempfile.mkdtemp()
        ds = torchvision.datasets.CelebA(root=temp_dir, download=True)
        assert len(ds) == 162770
        assert ds[40711] is not None

        # 2nd time, the ZIP file will be detected (because now it has been downloaded)
        ds2 = torchvision.datasets.CelebA(root=temp_dir, download=True)
        assert ds2.root_zip is not None, "Transparant ZIP reading support broken: ZIP file not found"
        assert len(ds2) == 162770
        assert ds2[40711] is not None
        shutil.rmtree(temp_dir)

    def test_omniglot(self):
        temp_dir = tempfile.mkdtemp()
        ds = torchvision.datasets.Omniglot(root=temp_dir, download=True)
        assert len(ds) == 19280
        assert ds[4071] is not None

        # 2nd time, the ZIP file will be detected (because now it has been downloaded)
        ds2 = torchvision.datasets.Omniglot(root=temp_dir, download=True)
        assert ds2.root_zip is not None, "Transparant ZIP reading support broken: ZIP file not found"
        assert len(ds2) == 19280
        assert ds2[4071] is not None
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
