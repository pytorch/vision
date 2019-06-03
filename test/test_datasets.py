import os
import shutil
import contextlib
import tempfile
import unittest
import mock
import PIL
import torchvision

FAKEDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'assets', 'fakedata')


@contextlib.contextmanager
def tmp_dir(src=None, **kwargs):
    tmp_dir = tempfile.mkdtemp(**kwargs)
    if src is not None:
        os.rmdir(tmp_dir)
        shutil.copytree(src, tmp_dir)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


class Tester(unittest.TestCase):

    def test_imagefolder(self):
        with tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
            classes = sorted(['a', 'b'])
            class_a_image_files = [os.path.join(root, 'a', file)
                                   for file in ('a1.png', 'a2.png', 'a3.png')]
            class_b_image_files = [os.path.join(root, 'b', file)
                                   for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')]
            dataset = torchvision.datasets.ImageFolder(root, loader=lambda x: x)

            # test if all classes are present
            self.assertEqual(classes, sorted(dataset.classes))

            # test if combination of classes and class_to_index functions correctly
            for cls in classes:
                self.assertEqual(cls, dataset.classes[dataset.class_to_idx[cls]])

            # test if all images were detected correctly
            class_a_idx = dataset.class_to_idx['a']
            class_b_idx = dataset.class_to_idx['b']
            imgs_a = [(img_file, class_a_idx) for img_file in class_a_image_files]
            imgs_b = [(img_file, class_b_idx) for img_file in class_b_image_files]
            imgs = sorted(imgs_a + imgs_b)
            self.assertEqual(imgs, dataset.imgs)

            # test if the datasets outputs all images correctly
            outputs = sorted([dataset[i] for i in range(len(dataset))])
            self.assertEqual(imgs, outputs)

            # redo all tests with specified valid image files
            dataset = torchvision.datasets.ImageFolder(root, loader=lambda x: x,
                                                       is_valid_file=lambda x: '3' in x)
            self.assertEqual(classes, sorted(dataset.classes))

            class_a_idx = dataset.class_to_idx['a']
            class_b_idx = dataset.class_to_idx['b']
            imgs_a = [(img_file, class_a_idx) for img_file in class_a_image_files
                      if '3' in img_file]
            imgs_b = [(img_file, class_b_idx) for img_file in class_b_image_files
                      if '3' in img_file]
            imgs = sorted(imgs_a + imgs_b)
            self.assertEqual(imgs, dataset.imgs)

            outputs = sorted([dataset[i] for i in range(len(dataset))])
            self.assertEqual(imgs, outputs)

    def test_mnist(self):
        with tmp_dir() as root:
            dataset = torchvision.datasets.MNIST(root, download=True)
            self.assertEqual(len(dataset), 60000)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))

    def test_kmnist(self):
        with tmp_dir() as root:
            dataset = torchvision.datasets.KMNIST(root, download=True)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))

    def test_fashionmnist(self):
        with tmp_dir() as root:
            dataset = torchvision.datasets.FashionMNIST(root, download=True)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))

    @mock.patch('torchvision.datasets.utils.download_url')
    def test_imagenet(self, mock_download):
        with tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagenet')) as root:
            dataset = torchvision.datasets.ImageNet(root, split='train', download=True)
            self.assertEqual(len(dataset), 3)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))
            self.assertEqual(dataset.class_to_idx['Tinca tinca'], target)

            dataset = torchvision.datasets.ImageNet(root, split='val', download=True)
            self.assertEqual(len(dataset), 3)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))
            self.assertTrue(isinstance(target, int))
            self.assertEqual(dataset.class_to_idx['Tinca tinca'], target)


if __name__ == '__main__':
    unittest.main()
