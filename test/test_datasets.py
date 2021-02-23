import contextlib
import sys
import os
import unittest
from unittest import mock
import numpy as np
import PIL
from PIL import Image
from torch._utils_internal import get_file_path_2
import torchvision
from torchvision.datasets import utils
from common_utils import get_tmp_dir
from fakedata_generation import mnist_root, cifar_root, imagenet_root, \
    cityscapes_root, svhn_root, ucf101_root, places365_root, widerface_root, stl10_root
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
import itertools
import datasets_utils
import pathlib
import pickle
from torchvision import datasets
import torch
import shutil
import json


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class DatasetTestcase(unittest.TestCase):
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


class Tester(DatasetTestcase):
    def test_imagefolder(self):
        # TODO: create the fake data on-the-fly
        FAKEDATA_DIR = get_file_path_2(
            os.path.dirname(os.path.abspath(__file__)), 'assets', 'fakedata')

        with get_tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
            classes = sorted(['a', 'b'])
            class_a_image_files = [
                os.path.join(root, 'a', file) for file in ('a1.png', 'a2.png', 'a3.png')
            ]
            class_b_image_files = [
                os.path.join(root, 'b', file) for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')
            ]
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
            dataset = torchvision.datasets.ImageFolder(
                root, loader=lambda x: x, is_valid_file=lambda x: '3' in x)
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

    def test_imagefolder_empty(self):
        with get_tmp_dir() as root:
            with self.assertRaises(RuntimeError):
                torchvision.datasets.ImageFolder(root, loader=lambda x: x)

            with self.assertRaises(RuntimeError):
                torchvision.datasets.ImageFolder(
                    root, loader=lambda x: x, is_valid_file=lambda x: False
                )

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

    @mock.patch('torchvision.datasets.imagenet._verify_archive')
    @unittest.skipIf(not HAS_SCIPY, "scipy unavailable")
    def test_imagenet(self, mock_verify):
        with imagenet_root() as root:
            dataset = torchvision.datasets.ImageNet(root, split='train')
            self.generic_classification_dataset_test(dataset)

            dataset = torchvision.datasets.ImageNet(root, split='val')
            self.generic_classification_dataset_test(dataset)

    @mock.patch('torchvision.datasets.WIDERFace._check_integrity')
    @unittest.skipIf('win' in sys.platform, 'temporarily disabled on Windows')
    def test_widerface(self, mock_check_integrity):
        mock_check_integrity.return_value = True
        with widerface_root() as root:
            dataset = torchvision.datasets.WIDERFace(root, split='train')
            self.assertEqual(len(dataset), 1)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))

            dataset = torchvision.datasets.WIDERFace(root, split='val')
            self.assertEqual(len(dataset), 1)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))

            dataset = torchvision.datasets.WIDERFace(root, split='test')
            self.assertEqual(len(dataset), 1)
            img, target = dataset[0]
            self.assertTrue(isinstance(img, PIL.Image.Image))

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
                        dataset = torchvision.datasets.Cityscapes(
                            root, split=split, target_type=target_type, mode=mode)
                        self.generic_segmentation_dataset_test(dataset, num_images=2)

                    color_dataset = torchvision.datasets.Cityscapes(
                        root, split=split, target_type='color', mode=mode)
                    color_img, color_target = color_dataset[0]
                    self.assertTrue(isinstance(color_img, PIL.Image.Image))
                    self.assertTrue(np.array(color_target).shape[2] == 4)

                    polygon_dataset = torchvision.datasets.Cityscapes(
                        root, split=split, target_type='polygon', mode=mode)
                    polygon_img, polygon_target = polygon_dataset[0]
                    self.assertTrue(isinstance(polygon_img, PIL.Image.Image))
                    self.assertTrue(isinstance(polygon_target, dict))
                    self.assertTrue(isinstance(polygon_target['imgHeight'], int))
                    self.assertTrue(isinstance(polygon_target['objects'], list))

                    # Test multiple target types
                    targets_combo = ['semantic', 'polygon', 'color']
                    multiple_types_dataset = torchvision.datasets.Cityscapes(
                        root, split=split, target_type=targets_combo, mode=mode)
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

    @unittest.skipIf(not HAS_PYAV, "PyAV unavailable")
    def test_ucf101(self):
        cached_meta_data = None
        with ucf101_root() as (root, ann_root):
            for split in {True, False}:
                for fold in range(1, 4):
                    for length in {10, 15, 20}:
                        dataset = torchvision.datasets.UCF101(root, ann_root, length, fold=fold, train=split,
                                                              num_workers=2, _precomputed_metadata=cached_meta_data)
                        if cached_meta_data is None:
                            cached_meta_data = dataset.metadata
                        self.assertGreater(len(dataset), 0)

                        video, audio, label = dataset[0]
                        self.assertEqual(video.size(), (length, 320, 240, 3))
                        self.assertEqual(audio.numel(), 0)
                        self.assertEqual(label, 0)

                        video, audio, label = dataset[len(dataset) - 1]
                        self.assertEqual(video.size(), (length, 320, 240, 3))
                        self.assertEqual(audio.numel(), 0)
                        self.assertEqual(label, 1)

    def test_places365(self):
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with places365_root(split=split, small=small) as places365:
                root, data = places365

                dataset = torchvision.datasets.Places365(root, split=split, small=small, download=True)
                self.generic_classification_dataset_test(dataset, num_images=len(data["imgs"]))

    def test_places365_transforms(self):
        expected_image = "image"
        expected_target = "target"

        def transform(image):
            return expected_image

        def target_transform(target):
            return expected_target

        with places365_root() as places365:
            root, data = places365

            dataset = torchvision.datasets.Places365(
                root, transform=transform, target_transform=target_transform, download=True
            )
            actual_image, actual_target = dataset[0]

            self.assertEqual(actual_image, expected_image)
            self.assertEqual(actual_target, expected_target)

    def test_places365_devkit_download(self):
        for split in ("train-standard", "train-challenge", "val"):
            with self.subTest(split=split):
                with places365_root(split=split) as places365:
                    root, data = places365

                    dataset = torchvision.datasets.Places365(root, split=split, download=True)

                    with self.subTest("classes"):
                        self.assertSequenceEqual(dataset.classes, data["classes"])

                    with self.subTest("class_to_idx"):
                        self.assertDictEqual(dataset.class_to_idx, data["class_to_idx"])

                    with self.subTest("imgs"):
                        self.assertSequenceEqual(dataset.imgs, data["imgs"])

    def test_places365_devkit_no_download(self):
        for split in ("train-standard", "train-challenge", "val"):
            with self.subTest(split=split):
                with places365_root(split=split) as places365:
                    root, data = places365

                    with self.assertRaises(RuntimeError):
                        torchvision.datasets.Places365(root, split=split, download=False)

    def test_places365_images_download(self):
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with self.subTest(split=split, small=small):
                with places365_root(split=split, small=small) as places365:
                    root, data = places365

                    dataset = torchvision.datasets.Places365(root, split=split, small=small, download=True)

                    assert all(os.path.exists(item[0]) for item in dataset.imgs)

    def test_places365_images_download_preexisting(self):
        split = "train-standard"
        small = False
        images_dir = "data_large_standard"

        with places365_root(split=split, small=small) as places365:
            root, data = places365
            os.mkdir(os.path.join(root, images_dir))

            with self.assertRaises(RuntimeError):
                torchvision.datasets.Places365(root, split=split, small=small, download=True)

    def test_places365_repr_smoke(self):
        with places365_root() as places365:
            root, data = places365

            dataset = torchvision.datasets.Places365(root, download=True)
            self.assertIsInstance(repr(dataset), str)


class STL10Tester(DatasetTestcase):
    @contextlib.contextmanager
    def mocked_root(self):
        with stl10_root() as (root, data):
            yield root, data

    @contextlib.contextmanager
    def mocked_dataset(self, pre_extract=False, download=True, **kwargs):
        with self.mocked_root() as (root, data):
            if pre_extract:
                utils.extract_archive(os.path.join(root, data["archive"]))
            dataset = torchvision.datasets.STL10(root, download=download, **kwargs)
            yield dataset, data

    def test_not_found(self):
        with self.assertRaises(RuntimeError):
            with self.mocked_dataset(download=False):
                pass

    def test_splits(self):
        for split in ('train', 'train+unlabeled', 'unlabeled', 'test'):
            with self.mocked_dataset(split=split) as (dataset, data):
                num_images = sum([data["num_images_in_split"][part] for part in split.split("+")])
                self.generic_classification_dataset_test(dataset, num_images=num_images)

    def test_folds(self):
        for fold in range(10):
            with self.mocked_dataset(split="train", folds=fold) as (dataset, data):
                num_images = data["num_images_in_folds"][fold]
                self.assertEqual(len(dataset), num_images)

    def test_invalid_folds1(self):
        with self.assertRaises(ValueError):
            with self.mocked_dataset(folds=10):
                pass

    def test_invalid_folds2(self):
        with self.assertRaises(ValueError):
            with self.mocked_dataset(folds="0"):
                pass

    def test_transforms(self):
        expected_image = "image"
        expected_target = "target"

        def transform(image):
            return expected_image

        def target_transform(target):
            return expected_target

        with self.mocked_dataset(transform=transform, target_transform=target_transform) as (dataset, _):
            actual_image, actual_target = dataset[0]

            self.assertEqual(actual_image, expected_image)
            self.assertEqual(actual_target, expected_target)

    def test_unlabeled(self):
        with self.mocked_dataset(split="unlabeled") as (dataset, _):
            labels = [dataset[idx][1] for idx in range(len(dataset))]
            self.assertTrue(all([label == -1 for label in labels]))

    @unittest.mock.patch("torchvision.datasets.stl10.download_and_extract_archive")
    def test_download_preexisting(self, mock):
        with self.mocked_dataset(pre_extract=True) as (dataset, data):
            mock.assert_not_called()

    def test_repr_smoke(self):
        with self.mocked_dataset() as (dataset, _):
            self.assertIsInstance(repr(dataset), str)


class Caltech101TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech101
    FEATURE_TYPES = (PIL.Image.Image, (int, np.ndarray, tuple))

    CONFIGS = datasets_utils.combinations_grid(target_type=("category", "annotation", ["category", "annotation"]))
    REQUIRED_PACKAGES = ("scipy",)

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "caltech101"
        images = root / "101_ObjectCategories"
        annotations = root / "Annotations"

        categories = (("Faces", "Faces_2"), ("helicopter", "helicopter"), ("ying_yang", "ying_yang"))
        num_images_per_category = 2

        for image_category, annotation_category in categories:
            datasets_utils.create_image_folder(
                root=images,
                name=image_category,
                file_name_fn=lambda idx: f"image_{idx + 1:04d}.jpg",
                num_examples=num_images_per_category,
            )
            self._create_annotation_folder(
                root=annotations,
                name=annotation_category,
                file_name_fn=lambda idx: f"annotation_{idx + 1:04d}.mat",
                num_examples=num_images_per_category,
            )

        # This is included in the original archive, but is removed by the dataset. Thus, an empty directory suffices.
        os.makedirs(images / "BACKGROUND_Google")

        return num_images_per_category * len(categories)

    def _create_annotation_folder(self, root, name, file_name_fn, num_examples):
        root = pathlib.Path(root) / name
        os.makedirs(root)

        for idx in range(num_examples):
            self._create_annotation_file(root, file_name_fn(idx))

    def _create_annotation_file(self, root, name):
        mdict = dict(obj_contour=torch.rand((2, torch.randint(3, 6, size=())), dtype=torch.float64).numpy())
        datasets_utils.lazy_importer.scipy.io.savemat(str(pathlib.Path(root) / name), mdict)

    def test_combined_targets(self):
        target_types = ["category", "annotation"]

        individual_targets = []
        for target_type in target_types:
            with self.create_dataset(target_type=target_type) as (dataset, _):
                _, target = dataset[0]
                individual_targets.append(target)

        with self.create_dataset(target_type=target_types) as (dataset, _):
            _, combined_targets = dataset[0]

        actual = len(individual_targets)
        expected = len(combined_targets)
        self.assertEqual(
            actual,
            expected,
            f"The number of the returned combined targets does not match the the number targets if requested "
            f"individually: {actual} != {expected}",
        )

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                self.assertIs(
                    actual,
                    expected,
                    f"Type of the combined target does not match the type of the corresponding individual target: "
                    f"{actual} is not {expected}",
                )


class Caltech256TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech256

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / "caltech256" / "256_ObjectCategories"

        categories = ((1, "ak47"), (127, "laptop-101"), (257, "clutter"))
        num_images_per_category = 2

        for idx, category in categories:
            datasets_utils.create_image_folder(
                tmpdir,
                name=f"{idx:03d}.{category}",
                file_name_fn=lambda image_idx: f"{idx:03d}_{image_idx + 1:04d}.jpg",
                num_examples=num_images_per_category,
            )

        return num_images_per_category * len(categories)


class CIFAR10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CIFAR10
    CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    _VERSION_CONFIG = dict(
        base_folder="cifar-10-batches-py",
        train_files=tuple(f"data_batch_{idx}" for idx in range(1, 6)),
        test_files=("test_batch",),
        labels_key="labels",
        meta_file="batches.meta",
        num_categories=10,
        categories_key="label_names",
    )

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / self._VERSION_CONFIG["base_folder"]
        os.makedirs(tmpdir)

        num_images_per_file = 1
        for name in itertools.chain(self._VERSION_CONFIG["train_files"], self._VERSION_CONFIG["test_files"]):
            self._create_batch_file(tmpdir, name, num_images_per_file)

        categories = self._create_meta_file(tmpdir)

        return dict(
            num_examples=num_images_per_file
            * len(self._VERSION_CONFIG["train_files"] if config["train"] else self._VERSION_CONFIG["test_files"]),
            categories=categories,
        )

    def _create_batch_file(self, root, name, num_images):
        data = datasets_utils.create_image_or_video_tensor((num_images, 32 * 32 * 3))
        labels = np.random.randint(0, self._VERSION_CONFIG["num_categories"], size=num_images).tolist()
        self._create_binary_file(root, name, {"data": data, self._VERSION_CONFIG["labels_key"]: labels})

    def _create_meta_file(self, root):
        categories = [
            f"{idx:0{len(str(self._VERSION_CONFIG['num_categories'] - 1))}d}"
            for idx in range(self._VERSION_CONFIG["num_categories"])
        ]
        self._create_binary_file(
            root, self._VERSION_CONFIG["meta_file"], {self._VERSION_CONFIG["categories_key"]: categories}
        )
        return categories

    def _create_binary_file(self, root, name, content):
        with open(pathlib.Path(root) / name, "wb") as fh:
            pickle.dump(content, fh)

    def test_class_to_idx(self):
        with self.create_dataset() as (dataset, info):
            expected = {category: label for label, category in enumerate(info["categories"])}
            actual = dataset.class_to_idx
            self.assertEqual(actual, expected)


class CIFAR100(CIFAR10TestCase):
    DATASET_CLASS = datasets.CIFAR100

    _VERSION_CONFIG = dict(
        base_folder="cifar-100-python",
        train_files=("train",),
        test_files=("test",),
        labels_key="fine_labels",
        meta_file="meta",
        num_categories=100,
        categories_key="fine_label_names",
    )


class CelebATestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CelebA
    FEATURE_TYPES = (PIL.Image.Image, (torch.Tensor, int, tuple, type(None)))

    CONFIGS = datasets_utils.combinations_grid(
        split=("train", "valid", "test", "all"),
        target_type=("attr", "identity", "bbox", "landmarks", ["attr", "identity"]),
    )
    REQUIRED_PACKAGES = ("pandas",)

    _SPLIT_TO_IDX = dict(train=0, valid=1, test=2)

    def inject_fake_data(self, tmpdir, config):
        base_folder = pathlib.Path(tmpdir) / "celeba"
        os.makedirs(base_folder)

        num_images, num_images_per_split = self._create_split_txt(base_folder)

        datasets_utils.create_image_folder(
            base_folder, "img_align_celeba", lambda idx: f"{idx + 1:06d}.jpg", num_images
        )
        attr_names = self._create_attr_txt(base_folder, num_images)
        self._create_identity_txt(base_folder, num_images)
        self._create_bbox_txt(base_folder, num_images)
        self._create_landmarks_txt(base_folder, num_images)

        return dict(num_examples=num_images_per_split[config["split"]], attr_names=attr_names)

    def _create_split_txt(self, root):
        num_images_per_split = dict(train=3, valid=2, test=1)

        data = [
            [self._SPLIT_TO_IDX[split]] for split, num_images in num_images_per_split.items() for _ in range(num_images)
        ]
        self._create_txt(root, "list_eval_partition.txt", data)

        num_images_per_split["all"] = num_images = sum(num_images_per_split.values())
        return num_images, num_images_per_split

    def _create_attr_txt(self, root, num_images):
        header = ("5_o_Clock_Shadow", "Young")
        data = torch.rand((num_images, len(header))).ge(0.5).int().mul(2).sub(1).tolist()
        self._create_txt(root, "list_attr_celeba.txt", data, header=header, add_num_examples=True)
        return header

    def _create_identity_txt(self, root, num_images):
        data = torch.randint(1, 4, size=(num_images, 1)).tolist()
        self._create_txt(root, "identity_CelebA.txt", data)

    def _create_bbox_txt(self, root, num_images):
        header = ("x_1", "y_1", "width", "height")
        data = torch.randint(10, size=(num_images, len(header))).tolist()
        self._create_txt(
            root, "list_bbox_celeba.txt", data, header=header, add_num_examples=True, add_image_id_to_header=True
        )

    def _create_landmarks_txt(self, root, num_images):
        header = ("lefteye_x", "rightmouth_y")
        data = torch.randint(10, size=(num_images, len(header))).tolist()
        self._create_txt(root, "list_landmarks_align_celeba.txt", data, header=header, add_num_examples=True)

    def _create_txt(self, root, name, data, header=None, add_num_examples=False, add_image_id_to_header=False):
        with open(pathlib.Path(root) / name, "w") as fh:
            if add_num_examples:
                fh.write(f"{len(data)}\n")

            if header:
                if add_image_id_to_header:
                    header = ("image_id", *header)
                fh.write(f"{' '.join(header)}\n")

            for idx, line in enumerate(data, 1):
                fh.write(f"{' '.join((f'{idx:06d}.jpg', *[str(value) for value in line]))}\n")

    def test_combined_targets(self):
        target_types = ["attr", "identity", "bbox", "landmarks"]

        individual_targets = []
        for target_type in target_types:
            with self.create_dataset(target_type=target_type) as (dataset, _):
                _, target = dataset[0]
                individual_targets.append(target)

        with self.create_dataset(target_type=target_types) as (dataset, _):
            _, combined_targets = dataset[0]

        actual = len(individual_targets)
        expected = len(combined_targets)
        self.assertEqual(
            actual,
            expected,
            f"The number of the returned combined targets does not match the the number targets if requested "
            f"individually: {actual} != {expected}",
        )

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                self.assertIs(
                    actual,
                    expected,
                    f"Type of the combined target does not match the type of the corresponding individual target: "
                    f"{actual} is not {expected}",
                )

    def test_no_target(self):
        with self.create_dataset(target_type=[]) as (dataset, _):
            _, target = dataset[0]

        self.assertIsNone(target)

    def test_attr_names(self):
        with self.create_dataset() as (dataset, info):
            self.assertEqual(tuple(dataset.attr_names), info["attr_names"])


class VOCSegmentationTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.VOCSegmentation
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image)

    CONFIGS = (
        *datasets_utils.combinations_grid(
            year=[f"20{year:02d}" for year in range(7, 13)], image_set=("train", "val", "trainval")
        ),
        dict(year="2007", image_set="test"),
        dict(year="2007-test", image_set="test"),
    )

    def inject_fake_data(self, tmpdir, config):
        year, is_test_set = (
            ("2007", True)
            if config["year"] == "2007-test" or config["image_set"] == "test"
            else (config["year"], False)
        )
        image_set = config["image_set"]

        base_dir = pathlib.Path(tmpdir)
        if year == "2011":
            base_dir /= "TrainVal"
        base_dir = base_dir / "VOCdevkit" / f"VOC{year}"
        os.makedirs(base_dir)

        num_images, num_images_per_image_set = self._create_image_set_files(base_dir, "ImageSets", is_test_set)
        datasets_utils.create_image_folder(base_dir, "JPEGImages", lambda idx: f"{idx:06d}.jpg", num_images)

        datasets_utils.create_image_folder(base_dir, "SegmentationClass", lambda idx: f"{idx:06d}.png", num_images)
        annotation = self._create_annotation_files(base_dir, "Annotations", num_images)

        return dict(num_examples=num_images_per_image_set[image_set], annotation=annotation)

    def _create_image_set_files(self, root, name, is_test_set):
        root = pathlib.Path(root) / name
        src = pathlib.Path(root) / "Main"
        os.makedirs(src, exist_ok=True)

        idcs = dict(train=(0, 1, 2), val=(3, 4), test=(5,))
        idcs["trainval"] = (*idcs["train"], *idcs["val"])

        for image_set in ("test",) if is_test_set else ("train", "val", "trainval"):
            self._create_image_set_file(src, image_set, idcs[image_set])

        shutil.copytree(src, root / "Segmentation")

        num_images = max(itertools.chain(*idcs.values())) + 1
        num_images_per_image_set = dict([(image_set, len(idcs_)) for image_set, idcs_ in idcs.items()])
        return num_images, num_images_per_image_set

    def _create_image_set_file(self, root, image_set, idcs):
        with open(pathlib.Path(root) / f"{image_set}.txt", "w") as fh:
            fh.writelines([f"{idx:06d}\n" for idx in idcs])

    def _create_annotation_files(self, root, name, num_images):
        root = pathlib.Path(root) / name
        os.makedirs(root)

        for idx in range(num_images):
            annotation = self._create_annotation_file(root, f"{idx:06d}.xml")

        return annotation

    def _create_annotation_file(self, root, name):
        def add_child(parent, name, text=None):
            child = ET.SubElement(parent, name)
            child.text = text
            return child

        def add_name(obj, name="dog"):
            add_child(obj, "name", name)
            return name

        def add_bndbox(obj, bndbox=None):
            if bndbox is None:
                bndbox = {"xmin": "1", "xmax": "2", "ymin": "3", "ymax": "4"}

            obj = add_child(obj, "bndbox")
            for name, text in bndbox.items():
                add_child(obj, name, text)

            return bndbox

        annotation = ET.Element("annotation")
        obj = add_child(annotation, "object")
        data = dict(name=add_name(obj), bndbox=add_bndbox(obj))

        with open(pathlib.Path(root) / name, "wb") as fh:
            fh.write(ET.tostring(annotation))

        return data


class VOCDetectionTestCase(VOCSegmentationTestCase):
    DATASET_CLASS = datasets.VOCDetection
    FEATURE_TYPES = (PIL.Image.Image, dict)

    def test_annotations(self):
        with self.create_dataset() as (dataset, info):
            _, target = dataset[0]

            self.assertIn("annotation", target)
            annotation = target["annotation"]

            self.assertIn("object", annotation)
            objects = annotation["object"]

            self.assertEqual(len(objects), 1)
            object = objects[0]

            self.assertEqual(object, info["annotation"])


class CocoDetectionTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CocoDetection
    FEATURE_TYPES = (PIL.Image.Image, list)

    REQUIRED_PACKAGES = ("pycocotools",)

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        num_images = 3
        num_annotations_per_image = 2

        image_folder = tmpdir / "images"
        files = datasets_utils.create_image_folder(
            tmpdir, name="images", file_name_fn=lambda idx: f"{idx:012d}.jpg", num_examples=num_images
        )
        file_names = [file.relative_to(image_folder) for file in files]

        annotation_folder = tmpdir / "annotations"
        os.makedirs(annotation_folder)
        annotation_file, info = self._create_annotation_file(annotation_folder, file_names, num_annotations_per_image)

        info["num_examples"] = num_images
        return (str(image_folder), str(annotation_file)), info

    def _create_annotation_file(self, root, file_names, num_annotations_per_image):
        image_ids = [int(file_name.stem) for file_name in file_names]
        images = [dict(file_name=str(file_name), id=id) for file_name, id in zip(file_names, image_ids)]

        annotations, info = self._create_annotations(image_ids, num_annotations_per_image)

        content = dict(images=images, annotations=annotations)
        return self._create_json(root, "annotations.json", content), info

    def _create_annotations(self, image_ids, num_annotations_per_image):
        annotations = datasets_utils.combinations_grid(
            image_id=image_ids, bbox=([1.0, 2.0, 3.0, 4.0],) * num_annotations_per_image
        )
        for id, annotation in enumerate(annotations):
            annotation["id"] = id
        return annotations, dict()

    def _create_json(self, root, name, content):
        file = pathlib.Path(root) / name
        with open(file, "w") as fh:
            json.dump(content, fh)
        return file


class CocoCaptionsTestCase(CocoDetectionTestCase):
    DATASET_CLASS = datasets.CocoCaptions

    def _create_annotations(self, image_ids, num_annotations_per_image):
        captions = [str(idx) for idx in range(num_annotations_per_image)]
        annotations = datasets_utils.combinations_grid(image_id=image_ids, caption=captions)
        for id, annotation in enumerate(annotations):
            annotation["id"] = id
        return annotations, dict(captions=captions)

    def test_captions(self):
        with self.create_dataset() as (dataset, info):
            _, captions = dataset[0]
            self.assertEqual(tuple(captions), tuple(info["captions"]))


if __name__ == "__main__":
    unittest.main()
