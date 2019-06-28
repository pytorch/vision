import os
import unittest
import mock
import PIL
from PIL import Image
from torch._utils_internal import get_file_path_2
from torch.utils.data import DataLoader
import torchvision
from common_utils import get_tmp_dir
from fakedata_generation import mnist_root, cifar_root, imagenet_root


class Tester(unittest.TestCase):
    def generic_classification_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))

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

    def test_tripletdataset(self):
        FAKEDATA_DIR = get_file_path_2(
            os.path.dirname(os.path.abspath(__file__)), 'assets', 'fakedata')

        with get_tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
            num_triplets = 1000
            class_a_image_files = [os.path.join(root, 'a', file)
                                   for file in ('a1.png', 'a2.png', 'a3.png')]
            class_b_image_files = [os.path.join(root, 'b', file)
                                   for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')]
            dset_f = torchvision.datasets.DatasetFolder(root, lambda x: x, extensions=('.png'))
            dataset = torchvision.datasets.TripletDataset(dset_f, lambda x: x, num_triplets)

            # test if all images were detected correctly
            class_a_idx = dset_f.class_to_idx['a']
            class_b_idx = dset_f.class_to_idx['b']
            self.assertEqual(class_a_image_files, dataset.class_samples[class_a_idx])
            self.assertEqual(class_b_image_files, dataset.class_samples[class_b_idx])

            # test if the dataset outputs triplets of correct form
            for triplet in dataset:
                anc, pos, neg = tuple(os.path.split(path)[1] for path in triplet)
                # must belong to same class
                self.assertEqual(anc[0], pos[0])

                # none should be same
                self.assertNotEqual(anc, pos)
                self.assertNotEqual(anc, neg)
                self.assertNotEqual(pos, neg)

            # Check with dataloader with more workers
            num_triplets = 101
            dataset = torchvision.datasets.TripletDataset(dset_f, lambda x: x, num_triplets)
            loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=lambda x: x)

            # Check that no samples are missed while splitting work
            num_samples = 0
            for triplet_batch in loader:
                num_samples += len(triplet_batch)

            self.assertEqual(num_samples, num_triplets)


if __name__ == '__main__':
    unittest.main()
