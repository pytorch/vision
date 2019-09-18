import sys
import os
import unittest
import mock
import numpy as np
import PIL
from PIL import Image
from torch._utils_internal import get_file_path_2
import torchvision
from common_utils import get_tmp_dir
from fakedata_generation import mnist_root, cifar_root, imagenet_root, \
    cityscapes_root, svhn_root


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Tester(unittest.TestCase):
    def generic_classification_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))

    def generic_segmentation_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, PIL.Image.Image))

    def test_imagefolder(self):
        # TODO: create the fake data on-the-fly
        FAKEDATA_DIR = get_file_path_2(
            os.path.dirname(os.path.abspath(__file__)), 'assets', 'fakedata')

        with get_tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
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

    def test_multiimagefolder(self):
        with get_tmp_dir() as root:
            # make fake dataset
            os.makedirs(os.path.join(root, 'a'))
            a = []
            for filename in ('a1.png', 'a2.png', 'a3.png'):
                result = Image.fromarray((np.random.rand(20, 20) * 255).astype(np.uint8))
                result.save(os.path.join(root, 'a', filename))
                a.append(os.path.join(root, 'a', filename))
            os.makedirs(os.path.join(root, 'b'))
            b = []
            for filename in ('b1.png', 'b2.png', 'b3.png'):
                result = Image.fromarray((np.random.rand(20, 20) * 128).astype(np.uint8))
                result.save(os.path.join(root, 'b', filename))
                b.append(os.path.join(root, 'b', filename))

            true_samples = list(zip(a, b))

            directories = [os.path.join(root, 'a'), os.path.join(root, 'b'), ]
            dataset = torchvision.datasets.MultiImageFolder(directories=directories, loader=lambda x: x)

            # test if all images were detected correctly and in the proper order
            self.assertEqual(len(true_samples), len(dataset.samples))
            for i, j in zip(true_samples, dataset.samples):
                self.assertEqual(i, j)

            # test if the datasets outputs all images correctly
            for i in range(len(dataset)):
                self.assertEqual([ *true_samples[i] ], dataset[i])

            # redo all tests with specified valid image files
            dataset = torchvision.datasets.MultiImageFolder(directories=directories, loader=lambda x: x,
                                                            is_valid_file=lambda x: '3.png' in x)
            true_samples = [true_samples[2]]

            # test if all images were detected correctly and in the proper order
            self.assertEqual(len(true_samples), len(dataset.samples))
            for i, j in zip(true_samples, dataset.samples):
                self.assertEqual(i, j)

            # test if the datasets outputs all images correctly
            for i in range(len(dataset)):
                self.assertEqual([ *true_samples[i] ], dataset[i])

    @mock.patch('torchvision.datasets.mnist.download_and_extract_archive')
    def test_mnist(self, mock_download_extract):
        num_examples = 30
        with mnist_root(num_examples, "MNIST") as root:
            dataset = torchvision.datasets.MNIST(root, download=True)
            self.generic_classification_dataset_test(dataset, num_images=num_examples)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

    @mock.patch('torchvision.datasets.mnist.download_and_extract_archive')
    def test_kmnist(self, mock_download_extract):
        num_examples = 30
        with mnist_root(num_examples, "KMNIST") as root:
            dataset = torchvision.datasets.KMNIST(root, download=True)
            self.generic_classification_dataset_test(dataset, num_images=num_examples)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

    @mock.patch('torchvision.datasets.mnist.download_and_extract_archive')
    def test_fashionmnist(self, mock_download_extract):
        num_examples = 30
        with mnist_root(num_examples, "FashionMNIST") as root:
            dataset = torchvision.datasets.FashionMNIST(root, download=True)
            self.generic_classification_dataset_test(dataset, num_images=num_examples)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

    @mock.patch('torchvision.datasets.utils.download_url')
    @unittest.skipIf(not HAS_SCIPY, "scipy unavailable")
    def test_imagenet(self, mock_download):
        with imagenet_root() as root:
            dataset = torchvision.datasets.ImageNet(root, split='train', download=True)
            self.generic_classification_dataset_test(dataset)

            dataset = torchvision.datasets.ImageNet(root, split='val', download=True)
            self.generic_classification_dataset_test(dataset)

    @mock.patch('torchvision.datasets.cifar.check_integrity')
    @mock.patch('torchvision.datasets.cifar.CIFAR10._check_integrity')
    def test_cifar10(self, mock_ext_check, mock_int_check):
        mock_ext_check.return_value = True
        mock_int_check.return_value = True
        with cifar_root('CIFAR10') as root:
            dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)
            self.generic_classification_dataset_test(dataset, num_images=5)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

            dataset = torchvision.datasets.CIFAR10(root, train=False, download=True)
            self.generic_classification_dataset_test(dataset)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

    @mock.patch('torchvision.datasets.cifar.check_integrity')
    @mock.patch('torchvision.datasets.cifar.CIFAR10._check_integrity')
    def test_cifar100(self, mock_ext_check, mock_int_check):
        mock_ext_check.return_value = True
        mock_int_check.return_value = True
        with cifar_root('CIFAR100') as root:
            dataset = torchvision.datasets.CIFAR100(root, train=True, download=True)
            self.generic_classification_dataset_test(dataset)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

            dataset = torchvision.datasets.CIFAR100(root, train=False, download=True)
            self.generic_classification_dataset_test(dataset)
            img, target = dataset[0]
            self.assertEqual(dataset.class_to_idx[dataset.classes[0]], target)

    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_cityscapes(self):
        with cityscapes_root() as root:

            for mode in ['coarse', 'fine']:

                if mode == 'coarse':
                    splits = ['train', 'train_extra', 'val']
                else:
                    splits = ['train', 'val', 'test']

                for split in splits:
                    for target_type in ['semantic', 'instance']:
                        dataset = torchvision.datasets.Cityscapes(root, split=split,
                                                                  target_type=target_type, mode=mode)
                        self.generic_segmentation_dataset_test(dataset, num_images=2)

                    color_dataset = torchvision.datasets.Cityscapes(root, split=split,
                                                                    target_type='color', mode=mode)
                    color_img, color_target = color_dataset[0]
                    self.assertTrue(isinstance(color_img, PIL.Image.Image))
                    self.assertTrue(np.array(color_target).shape[2] == 4)

                    polygon_dataset = torchvision.datasets.Cityscapes(root, split=split,
                                                                      target_type='polygon', mode=mode)
                    polygon_img, polygon_target = polygon_dataset[0]
                    self.assertTrue(isinstance(polygon_img, PIL.Image.Image))
                    self.assertTrue(isinstance(polygon_target, dict))
                    self.assertTrue(isinstance(polygon_target['imgHeight'], int))
                    self.assertTrue(isinstance(polygon_target['objects'], list))

                    # Test multiple target types
                    targets_combo = ['semantic', 'polygon', 'color']
                    multiple_types_dataset = torchvision.datasets.Cityscapes(root, split=split,
                                                                             target_type=targets_combo,
                                                                             mode=mode)
                    output = multiple_types_dataset[0]
                    self.assertTrue(isinstance(output, tuple))
                    self.assertTrue(len(output) == 2)
                    self.assertTrue(isinstance(output[0], PIL.Image.Image))
                    self.assertTrue(isinstance(output[1], tuple))
                    self.assertTrue(len(output[1]) == 3)
                    self.assertTrue(isinstance(output[1][0], PIL.Image.Image))  # semantic
                    self.assertTrue(isinstance(output[1][1], dict))  # polygon
                    self.assertTrue(isinstance(output[1][2], PIL.Image.Image))  # color

    @mock.patch('torchvision.datasets.SVHN._check_integrity')
    @unittest.skipIf(not HAS_SCIPY, "scipy unavailable")
    def test_svhn(self, mock_check):
        mock_check.return_value = True
        with svhn_root() as root:
            dataset = torchvision.datasets.SVHN(root, split="train")
            self.generic_classification_dataset_test(dataset, num_images=2)

            dataset = torchvision.datasets.SVHN(root, split="test")
            self.generic_classification_dataset_test(dataset, num_images=2)

            dataset = torchvision.datasets.SVHN(root, split="extra")
            self.generic_classification_dataset_test(dataset, num_images=2)


if __name__ == '__main__':
    unittest.main()
