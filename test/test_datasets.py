import os
import shutil
import tempfile
import unittest
import mock
import PIL
import torchvision

FAKEDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'assets', 'fakedata')


class Tester(unittest.TestCase):

    # def test_mnist(self):
    #     tmp_dir = tempfile.mkdtemp()
    #     dataset = torchvision.datasets.MNIST(tmp_dir, download=True)
    #     self.assertEqual(len(dataset), 60000)
    #     img, target = dataset[0]
    #     self.assertTrue(isinstance(img, PIL.Image.Image))
    #     self.assertTrue(isinstance(target, int))
    #     shutil.rmtree(tmp_dir)
    #
    # def test_kmnist(self):
    #     tmp_dir = tempfile.mkdtemp()
    #     dataset = torchvision.datasets.KMNIST(tmp_dir, download=True)
    #     img, target = dataset[0]
    #     self.assertTrue(isinstance(img, PIL.Image.Image))
    #     self.assertTrue(isinstance(target, int))
    #     shutil.rmtree(tmp_dir)
    #
    # def test_fashionmnist(self):
    #     tmp_dir = tempfile.mkdtemp()
    #     dataset = torchvision.datasets.FashionMNIST(tmp_dir, download=True)
    #     img, target = dataset[0]
    #     self.assertTrue(isinstance(img, PIL.Image.Image))
    #     self.assertTrue(isinstance(target, int))
    #     shutil.rmtree(tmp_dir)

    @mock.patch('torchvision.datasets.utils.download_url')
    def test_imagenet(self, mock_download):
        tmp_dir = tempfile.mkdtemp()
        archives = ('ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar',
                    'ILSVRC2012_devkit_t12.tar.gz')
        for archive in archives:
            shutil.copy(os.path.join(FAKEDATA_DIR, 'imagenet', archive),
                        os.path.join(tmp_dir, archive))

        dataset = torchvision.datasets.ImageNet(tmp_dir, split='train', download=True)
        self.assertEqual(len(dataset), 3)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        self.assertEqual(dataset.class_to_idx['Tinca tinca'], target)

        dataset = torchvision.datasets.ImageNet(tmp_dir, split='val', download=True)
        self.assertEqual(len(dataset), 3)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))
        self.assertEqual(dataset.class_to_idx['Tinca tinca'], target)

        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()
