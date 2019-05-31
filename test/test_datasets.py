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

    def test_imagenet(self):
        tmp_dir = tempfile.mkdtemp()
        archive_dict = torchvision.datasets.imagenet.ARCHIVE_DICT
        archive_dict['train']['url'] = 'https://github.com/pmeier/vision/blob/imagenet_test/test/assets/dataset/fakedata/imagenet/ILSVRC2012_img_train.tar'
        archive_dict['val']['url'] = 'https://github.com/pmeier/vision/blob/imagenet_test/test/assets/dataset/fakedata/imagenet/ILSVRC2012_img_val.tar'

        dataset_train = torchvision.datasets.ImageNet(tmp_dir, split='train', download=True)
        self.assertEqual(len(dataset_train), 3)
        img, target = dataset_train[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))

        dataset_val = torchvision.datasets.ImageNet(tmp_dir, split='val', download=True)
        self.assertEqual(len(dataset_val), 3)
        img, target = dataset_val[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))

        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()
