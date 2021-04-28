import bz2
import contextlib
import io
import itertools
import os
import pathlib
import pickle
import json
import random
import shutil
import string
import unittest
import xml.etree.ElementTree as ET
import zipfile

import PIL
import datasets_utils
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets


class STL10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.STL10
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("train", "test", "unlabeled", "train+unlabeled"))

    @staticmethod
    def _make_binary_file(num_elements, root, name):
        file_name = os.path.join(root, name)
        np.zeros(num_elements, dtype=np.uint8).tofile(file_name)

    @staticmethod
    def _make_image_file(num_images, root, name, num_channels=3, height=96, width=96):
        STL10TestCase._make_binary_file(num_images * num_channels * height * width, root, name)

    @staticmethod
    def _make_label_file(num_images, root, name):
        STL10TestCase._make_binary_file(num_images, root, name)

    @staticmethod
    def _make_class_names_file(root, name="class_names.txt"):
        with open(os.path.join(root, name), "w") as fh:
            for cname in ("airplane", "bird"):
                fh.write(f"{cname}\n")

    @staticmethod
    def _make_fold_indices_file(root):
        num_folds = 10
        offset = 0
        with open(os.path.join(root, "fold_indices.txt"), "w") as fh:
            for fold in range(num_folds):
                line = " ".join([str(idx) for idx in range(offset, offset + fold + 1)])
                fh.write(f"{line}\n")
                offset += fold + 1

        return tuple(range(1, num_folds + 1))

    @staticmethod
    def _make_train_files(root, num_unlabeled_images=1):
        num_images_in_fold = STL10TestCase._make_fold_indices_file(root)
        num_train_images = sum(num_images_in_fold)

        STL10TestCase._make_image_file(num_train_images, root, "train_X.bin")
        STL10TestCase._make_label_file(num_train_images, root, "train_y.bin")
        STL10TestCase._make_image_file(1, root, "unlabeled_X.bin")

        return dict(train=num_train_images, unlabeled=num_unlabeled_images)

    @staticmethod
    def _make_test_files(root, num_images=2):
        STL10TestCase._make_image_file(num_images, root, "test_X.bin")
        STL10TestCase._make_label_file(num_images, root, "test_y.bin")

        return dict(test=num_images)

    def inject_fake_data(self, tmpdir, config):
        root_folder = os.path.join(tmpdir, "stl10_binary")
        os.mkdir(root_folder)

        num_images_in_split = self._make_train_files(root_folder)
        num_images_in_split.update(self._make_test_files(root_folder))
        self._make_class_names_file(root_folder)

        return sum(num_images_in_split[part] for part in config["split"].split("+"))

    def test_folds(self):
        for fold in range(10):
            with self.create_dataset(split="train", folds=fold) as (dataset, _):
                self.assertEqual(len(dataset), fold + 1)

    def test_unlabeled(self):
        with self.create_dataset(split="unlabeled") as (dataset, _):
            labels = [dataset[idx][1] for idx in range(len(dataset))]
            self.assertTrue(all(label == -1 for label in labels))

    def test_invalid_folds1(self):
        with self.assertRaises(ValueError):
            with self.create_dataset(folds=10):
                pass

    def test_invalid_folds2(self):
        with self.assertRaises(ValueError):
            with self.create_dataset(folds="0"):
                pass


class Caltech101TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech101
    FEATURE_TYPES = (PIL.Image.Image, (int, np.ndarray, tuple))

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        target_type=("category", "annotation", ["category", "annotation"])
    )
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


class WIDERFaceTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.WIDERFace
    FEATURE_TYPES = (PIL.Image.Image, (dict, type(None)))  # test split returns None as target
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(split=('train', 'val', 'test'))

    def inject_fake_data(self, tmpdir, config):
        widerface_dir = pathlib.Path(tmpdir) / 'widerface'
        annotations_dir = widerface_dir / 'wider_face_split'
        os.makedirs(annotations_dir)

        split_to_idx = split_to_num_examples = {
            "train": 1,
            "val": 2,
            "test": 3,
        }

        # We need to create all folders regardless of the split in config
        for split in ('train', 'val', 'test'):
            split_idx = split_to_idx[split]
            num_examples = split_to_num_examples[split]

            datasets_utils.create_image_folder(
                root=tmpdir,
                name=widerface_dir / f'WIDER_{split}' / 'images' / '0--Parade',
                file_name_fn=lambda image_idx: f"0_Parade_marchingband_1_{split_idx + image_idx}.jpg",
                num_examples=num_examples,
            )

            annotation_file_name = {
                'train': annotations_dir / 'wider_face_train_bbx_gt.txt',
                'val': annotations_dir / 'wider_face_val_bbx_gt.txt',
                'test': annotations_dir / 'wider_face_test_filelist.txt',
            }[split]

            annotation_content = {
                "train": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n1\n449 330 122 149 0 0 0 0 0 0\n"
                    for image_idx in range(num_examples)
                ),
                "val": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n1\n501 160 285 443 0 0 0 0 0 0\n"
                    for image_idx in range(num_examples)
                ),
                "test": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n"
                    for image_idx in range(num_examples)
                ),
            }[split]

            with open(annotation_file_name, "w") as annotation_file:
                annotation_file.write(annotation_content)

        return split_to_num_examples[config["split"]]


class CityScapesTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Cityscapes
    TARGET_TYPES = (
        "instance",
        "semantic",
        "polygon",
        "color",
    )
    ADDITIONAL_CONFIGS = (
        *datasets_utils.combinations_grid(
            mode=("fine",), split=("train", "test", "val"), target_type=TARGET_TYPES
        ),
        *datasets_utils.combinations_grid(
            mode=("coarse",),
            split=("train", "train_extra", "val"),
            target_type=TARGET_TYPES,
        ),
    )
    FEATURE_TYPES = (PIL.Image.Image, (dict, PIL.Image.Image))

    def inject_fake_data(self, tmpdir, config):

        tmpdir = pathlib.Path(tmpdir)

        mode_to_splits = {
            "Coarse": ["train", "train_extra", "val"],
            "Fine": ["train", "test", "val"],
        }

        if config["split"] == "train":  # just for coverage of the number of samples
            cities = ["bochum", "bremen"]
        else:
            cities = ["bochum"]

        polygon_target = {
            "imgHeight": 1024,
            "imgWidth": 2048,
            "objects": [
                {
                    "label": "sky",
                    "polygon": [
                        [1241, 0],
                        [1234, 156],
                        [1478, 197],
                        [1611, 172],
                        [1606, 0],
                    ],
                },
                {
                    "label": "road",
                    "polygon": [
                        [0, 448],
                        [1331, 274],
                        [1473, 265],
                        [2047, 605],
                        [2047, 1023],
                        [0, 1023],
                    ],
                },
            ],
        }

        for mode in ["Coarse", "Fine"]:
            gt_dir = tmpdir / f"gt{mode}"
            for split in mode_to_splits[mode]:
                for city in cities:
                    def make_image(name, size=10):
                        datasets_utils.create_image_folder(
                            root=gt_dir / split,
                            name=city,
                            file_name_fn=lambda _: name,
                            size=size,
                            num_examples=1,
                        )
                    make_image(f"{city}_000000_000000_gt{mode}_instanceIds.png")
                    make_image(f"{city}_000000_000000_gt{mode}_labelIds.png")
                    make_image(f"{city}_000000_000000_gt{mode}_color.png", size=(4, 10, 10))

                    polygon_target_name = gt_dir / split / city / f"{city}_000000_000000_gt{mode}_polygons.json"
                    with open(polygon_target_name, "w") as outfile:
                        json.dump(polygon_target, outfile)

        # Create leftImg8bit folder
        for split in ['test', 'train_extra', 'train', 'val']:
            for city in cities:
                datasets_utils.create_image_folder(
                    root=tmpdir / "leftImg8bit" / split,
                    name=city,
                    file_name_fn=lambda _: f"{city}_000000_000000_leftImg8bit.png",
                    num_examples=1,
                )

        info = {'num_examples': len(cities)}
        if config['target_type'] == 'polygon':
            info['expected_polygon_target'] = polygon_target
        return info

    def test_combined_targets(self):
        target_types = ['semantic', 'polygon', 'color']

        with self.create_dataset(target_type=target_types) as (dataset, _):
            output = dataset[0]
            self.assertTrue(isinstance(output, tuple))
            self.assertTrue(len(output) == 2)
            self.assertTrue(isinstance(output[0], PIL.Image.Image))
            self.assertTrue(isinstance(output[1], tuple))
            self.assertTrue(len(output[1]) == 3)
            self.assertTrue(isinstance(output[1][0], PIL.Image.Image))  # semantic
            self.assertTrue(isinstance(output[1][1], dict))  # polygon
            self.assertTrue(isinstance(output[1][2], PIL.Image.Image))  # color

    def test_feature_types_target_color(self):
        with self.create_dataset(target_type='color') as (dataset, _):
            color_img, color_target = dataset[0]
            self.assertTrue(isinstance(color_img, PIL.Image.Image))
            self.assertTrue(np.array(color_target).shape[2] == 4)

    def test_feature_types_target_polygon(self):
        with self.create_dataset(target_type='polygon') as (dataset, info):
            polygon_img, polygon_target = dataset[0]
            self.assertTrue(isinstance(polygon_img, PIL.Image.Image))
            self.assertEqual(polygon_target, info['expected_polygon_target'])


class ImageNetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ImageNet
    REQUIRED_PACKAGES = ('scipy',)
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(split=('train', 'val'))

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        wnid = 'n01234567'
        if config['split'] == 'train':
            num_examples = 3
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / 'train' / wnid / wnid,
                file_name_fn=lambda image_idx: f"{wnid}_{image_idx}.JPEG",
                num_examples=num_examples,
            )
        else:
            num_examples = 1
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / 'val' / wnid,
                file_name_fn=lambda image_ifx: "ILSVRC2012_val_0000000{image_idx}.JPEG",
                num_examples=num_examples,
            )

        wnid_to_classes = {wnid: [1]}
        torch.save((wnid_to_classes, None), tmpdir / 'meta.bin')
        return num_examples


class CIFAR10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CIFAR10
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

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

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("train", "valid", "test", "all"),
        target_type=("attr", "identity", "bbox", "landmarks", ["attr", "identity"]),
    )

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

    ADDITIONAL_CONFIGS = (
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

    _IMAGE_FOLDER = "images"
    _ANNOTATIONS_FOLDER = "annotations"
    _ANNOTATIONS_FILE = "annotations.json"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._IMAGE_FOLDER
        annotation_file = tmpdir / self._ANNOTATIONS_FOLDER / self._ANNOTATIONS_FILE
        return root, annotation_file

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        num_images = 3
        num_annotations_per_image = 2

        files = datasets_utils.create_image_folder(
            tmpdir, name=self._IMAGE_FOLDER, file_name_fn=lambda idx: f"{idx:012d}.jpg", num_examples=num_images
        )
        file_names = [file.relative_to(tmpdir / self._IMAGE_FOLDER) for file in files]

        annotation_folder = tmpdir / self._ANNOTATIONS_FOLDER
        os.makedirs(annotation_folder)
        info = self._create_annotation_file(
            annotation_folder, self._ANNOTATIONS_FILE, file_names, num_annotations_per_image
        )

        info["num_examples"] = num_images
        return info

    def _create_annotation_file(self, root, name, file_names, num_annotations_per_image):
        image_ids = [int(file_name.stem) for file_name in file_names]
        images = [dict(file_name=str(file_name), id=id) for file_name, id in zip(file_names, image_ids)]

        annotations, info = self._create_annotations(image_ids, num_annotations_per_image)
        self._create_json(root, name, dict(images=images, annotations=annotations))

        return info

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


class UCF101TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.UCF101

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(fold=(1, 2, 3), train=(True, False))

    _VIDEO_FOLDER = "videos"
    _ANNOTATIONS_FOLDER = "annotations"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._VIDEO_FOLDER
        annotation_path = tmpdir / self._ANNOTATIONS_FOLDER
        return root, annotation_path

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        video_folder = tmpdir / self._VIDEO_FOLDER
        os.makedirs(video_folder)
        video_files = self._create_videos(video_folder)

        annotations_folder = tmpdir / self._ANNOTATIONS_FOLDER
        os.makedirs(annotations_folder)
        num_examples = self._create_annotation_files(annotations_folder, video_files, config["fold"], config["train"])

        return num_examples

    def _create_videos(self, root, num_examples_per_class=3):
        def file_name_fn(cls, idx, clips_per_group=2):
            return f"v_{cls}_g{(idx // clips_per_group) + 1:02d}_c{(idx % clips_per_group) + 1:02d}.avi"

        video_files = [
            datasets_utils.create_video_folder(root, cls, lambda idx: file_name_fn(cls, idx), num_examples_per_class)
            for cls in ("ApplyEyeMakeup", "YoYo")
        ]
        return [path.relative_to(root) for path in itertools.chain(*video_files)]

    def _create_annotation_files(self, root, video_files, fold, train):
        current_videos = random.sample(video_files, random.randrange(1, len(video_files) - 1))
        current_annotation = self._annotation_file_name(fold, train)
        self._create_annotation_file(root, current_annotation, current_videos)

        other_videos = set(video_files) - set(current_videos)
        other_annotations = [
            self._annotation_file_name(fold, train) for fold, train in itertools.product((1, 2, 3), (True, False))
        ]
        other_annotations.remove(current_annotation)
        for name in other_annotations:
            self._create_annotation_file(root, name, other_videos)

        return len(current_videos)

    def _annotation_file_name(self, fold, train):
        return f"{'train' if train else 'test'}list{fold:02d}.txt"

    def _create_annotation_file(self, root, name, video_files):
        with open(pathlib.Path(root) / name, "w") as fh:
            fh.writelines(f"{file}\n" for file in sorted(video_files))


class LSUNTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.LSUN

    REQUIRED_PACKAGES = ("lmdb",)
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        classes=("train", "test", "val", ["bedroom_train", "church_outdoor_train"])
    )

    _CATEGORIES = (
        "bedroom",
        "bridge",
        "church_outdoor",
        "classroom",
        "conference_room",
        "dining_room",
        "kitchen",
        "living_room",
        "restaurant",
        "tower",
    )

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir)

        num_images = 0
        for cls in self._parse_classes(config["classes"]):
            num_images += self._create_lmdb(root, cls)

        return num_images

    @contextlib.contextmanager
    def create_dataset(
        self,
        *args, **kwargs
    ):
        with super().create_dataset(*args, **kwargs) as output:
            yield output
            # Currently datasets.LSUN caches the keys in the current directory rather than in the root directory. Thus,
            # this creates a number of _cache_* files in the current directory that will not be removed together
            # with the temporary directory
            for file in os.listdir(os.getcwd()):
                if file.startswith("_cache_"):
                    try:
                        os.remove(file)
                    except FileNotFoundError:
                        # When the same test is run in parallel (in fb internal tests), a thread may remove another
                        # thread's file. We should be able to remove the try/except when
                        # https://github.com/pytorch/vision/issues/825 is fixed.
                        pass

    def _parse_classes(self, classes):
        if not isinstance(classes, str):
            return classes

        split = classes
        if split == "test":
            return [split]

        return [f"{category}_{split}" for category in self._CATEGORIES]

    def _create_lmdb(self, root, cls):
        lmdb = datasets_utils.lazy_importer.lmdb
        hexdigits_lowercase = string.digits + string.ascii_lowercase[:6]

        folder = f"{cls}_lmdb"

        num_images = torch.randint(1, 4, size=()).item()
        format = "png"
        files = datasets_utils.create_image_folder(root, folder, lambda idx: f"{idx}.{format}", num_images)

        with lmdb.open(str(root / folder)) as env, env.begin(write=True) as txn:
            for file in files:
                key = "".join(random.choice(hexdigits_lowercase) for _ in range(40)).encode()

                buffer = io.BytesIO()
                PIL.Image.open(file).save(buffer, format)
                buffer.seek(0)
                value = buffer.read()

                txn.put(key, value)

                os.remove(file)

        return num_images

    def test_not_found_or_corrupted(self):
        # LSUN does not raise built-in exception, but a custom one. It is expressive enough to not 'cast' it to
        # RuntimeError or FileNotFoundError that are normally checked by this test.
        with self.assertRaises(datasets_utils.lazy_importer.lmdb.Error):
            super().test_not_found_or_corrupted()


class Kinetics400TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.Kinetics400

    def inject_fake_data(self, tmpdir, config):
        classes = ("Abseiling", "Zumba")
        num_videos_per_class = 2

        digits = string.ascii_letters + string.digits + "-_"
        for cls in classes:
            datasets_utils.create_video_folder(
                tmpdir,
                cls,
                lambda _: f"{datasets_utils.create_random_string(11, digits)}.avi",
                num_videos_per_class,
            )

        return num_videos_per_class * len(classes)


class HMDB51TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.HMDB51

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(fold=(1, 2, 3), train=(True, False))

    _VIDEO_FOLDER = "videos"
    _SPLITS_FOLDER = "splits"
    _CLASSES = ("brush_hair", "wave")

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._VIDEO_FOLDER
        annotation_path = tmpdir / self._SPLITS_FOLDER
        return root, annotation_path

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        video_folder = tmpdir / self._VIDEO_FOLDER
        os.makedirs(video_folder)
        video_files = self._create_videos(video_folder)

        splits_folder = tmpdir / self._SPLITS_FOLDER
        os.makedirs(splits_folder)
        num_examples = self._create_split_files(splits_folder, video_files, config["fold"], config["train"])

        return num_examples

    def _create_videos(self, root, num_examples_per_class=3):
        def file_name_fn(cls, idx, clips_per_group=2):
            return f"{cls}_{(idx // clips_per_group) + 1:d}_{(idx % clips_per_group) + 1:d}.avi"

        return [
            (
                cls,
                datasets_utils.create_video_folder(
                    root,
                    cls,
                    lambda idx: file_name_fn(cls, idx),
                    num_examples_per_class,
                ),
            )
            for cls in self._CLASSES
        ]

    def _create_split_files(self, root, video_files, fold, train):
        num_videos = num_train_videos = 0

        for cls, videos in video_files:
            num_videos += len(videos)

            train_videos = set(random.sample(videos, random.randrange(1, len(videos) - 1)))
            num_train_videos += len(train_videos)

            with open(pathlib.Path(root) / f"{cls}_test_split{fold}.txt", "w") as fh:
                fh.writelines(f"{file.name} {1 if file in train_videos else 2}\n" for file in videos)

        return num_train_videos if train else (num_videos - num_train_videos)


class OmniglotTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Omniglot

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(background=(True, False))

    def inject_fake_data(self, tmpdir, config):
        target_folder = (
            pathlib.Path(tmpdir) / "omniglot-py" / f"images_{'background' if config['background'] else 'evaluation'}"
        )
        os.makedirs(target_folder)

        num_images = 0
        for name in ("Alphabet_of_the_Magi", "Tifinagh"):
            num_images += self._create_alphabet_folder(target_folder, name)

        return num_images

    def _create_alphabet_folder(self, root, name):
        num_images_total = 0
        for idx in range(torch.randint(1, 4, size=()).item()):
            num_images = torch.randint(1, 4, size=()).item()
            num_images_total += num_images

            datasets_utils.create_image_folder(
                root / name, f"character{idx:02d}", lambda image_idx: f"{image_idx:02d}.png", num_images
            )

        return num_images_total


class SBUTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SBU
    FEATURE_TYPES = (PIL.Image.Image, str)

    def inject_fake_data(self, tmpdir, config):
        num_images = 3

        dataset_folder = pathlib.Path(tmpdir) / "dataset"
        images = datasets_utils.create_image_folder(tmpdir, "dataset", self._create_file_name, num_images)

        self._create_urls_txt(dataset_folder, images)
        self._create_captions_txt(dataset_folder, num_images)

        return num_images

    def _create_file_name(self, idx):
        part1 = datasets_utils.create_random_string(10, string.digits)
        part2 = datasets_utils.create_random_string(10, string.ascii_lowercase, string.digits[:6])
        return f"{part1}_{part2}.jpg"

    def _create_urls_txt(self, root, images):
        with open(root / "SBU_captioned_photo_dataset_urls.txt", "w") as fh:
            for image in images:
                fh.write(
                    f"http://static.flickr.com/{datasets_utils.create_random_string(4, string.digits)}/{image.name}\n"
                )

    def _create_captions_txt(self, root, num_images):
        with open(root / "SBU_captioned_photo_dataset_captions.txt", "w") as fh:
            for _ in range(num_images):
                fh.write(f"{datasets_utils.create_random_string(10)}\n")


class SEMEIONTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SEMEION

    def inject_fake_data(self, tmpdir, config):
        num_images = 3

        images = torch.rand(num_images, 256)
        labels = F.one_hot(torch.randint(10, size=(num_images,)))
        with open(pathlib.Path(tmpdir) / "semeion.data", "w") as fh:
            for image, one_hot_labels in zip(images, labels):
                image_columns = " ".join([f"{pixel.item():.4f}" for pixel in image])
                labels_columns = " ".join([str(label.item()) for label in one_hot_labels])
                fh.write(f"{image_columns} {labels_columns}\n")

        return num_images


class USPSTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.USPS

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    def inject_fake_data(self, tmpdir, config):
        num_images = 2 if config["train"] else 1

        images = torch.rand(num_images, 256) * 2 - 1
        labels = torch.randint(1, 11, size=(num_images,))

        with bz2.open(pathlib.Path(tmpdir) / f"usps{'.t' if not config['train'] else ''}.bz2", "w") as fh:
            for image, label in zip(images, labels):
                line = " ".join((str(label.item()), *[f"{idx}:{pixel:.6f}" for idx, pixel in enumerate(image, 1)]))
                fh.write(f"{line}\n".encode())

        return num_images


class SBDatasetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SBDataset
    FEATURE_TYPES = (PIL.Image.Image, (np.ndarray, PIL.Image.Image))

    REQUIRED_PACKAGES = ("scipy.io", "scipy.sparse")

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        image_set=("train", "val", "train_noval"), mode=("boundaries", "segmentation")
    )

    _NUM_CLASSES = 20

    def inject_fake_data(self, tmpdir, config):
        num_images, num_images_per_image_set = self._create_split_files(tmpdir)

        sizes = self._create_target_folder(tmpdir, "cls", num_images)

        datasets_utils.create_image_folder(
            tmpdir, "img", lambda idx: f"{self._file_stem(idx)}.jpg", num_images, size=lambda idx: sizes[idx]
        )

        return num_images_per_image_set[config["image_set"]]

    def _create_split_files(self, root):
        root = pathlib.Path(root)

        splits = dict(train=(0, 1, 2), train_noval=(0, 2), val=(3,))

        for split, idcs in splits.items():
            self._create_split_file(root, split, idcs)

        num_images = max(itertools.chain(*splits.values())) + 1
        num_images_per_split = dict([(split, len(idcs)) for split, idcs in splits.items()])
        return num_images, num_images_per_split

    def _create_split_file(self, root, name, idcs):
        with open(root / f"{name}.txt", "w") as fh:
            fh.writelines(f"{self._file_stem(idx)}\n" for idx in idcs)

    def _create_target_folder(self, root, name, num_images):
        io = datasets_utils.lazy_importer.scipy.io

        target_folder = pathlib.Path(root) / name
        os.makedirs(target_folder)

        sizes = [torch.randint(1, 4, size=(2,)).tolist() for _ in range(num_images)]
        for idx, size in enumerate(sizes):
            content = dict(
                GTcls=dict(Boundaries=self._create_boundaries(size), Segmentation=self._create_segmentation(size))
            )
            io.savemat(target_folder / f"{self._file_stem(idx)}.mat", content)

        return sizes

    def _create_boundaries(self, size):
        sparse = datasets_utils.lazy_importer.scipy.sparse
        return [
            [sparse.csc_matrix(torch.randint(0, 2, size=size, dtype=torch.uint8).numpy())]
            for _ in range(self._NUM_CLASSES)
        ]

    def _create_segmentation(self, size):
        return torch.randint(0, self._NUM_CLASSES + 1, size=size, dtype=torch.uint8).numpy()

    def _file_stem(self, idx):
        return f"2008_{idx:06d}"


class FakeDataTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FakeData
    FEATURE_TYPES = (PIL.Image.Image, int)

    def dataset_args(self, tmpdir, config):
        return ()

    def inject_fake_data(self, tmpdir, config):
        return config["size"]

    def test_not_found_or_corrupted(self):
        self.skipTest("The data is generated at creation and thus cannot be non-existent or corrupted.")


class PhotoTourTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.PhotoTour

    # The PhotoTour dataset returns examples with different features with respect to the 'train' parameter. Thus,
    # we overwrite 'FEATURE_TYPES' with a dummy value to satisfy the initial checks of the base class. Furthermore, we
    # overwrite the 'test_feature_types()' method to select the correct feature types before the test is run.
    FEATURE_TYPES = ()
    _TRAIN_FEATURE_TYPES = (torch.Tensor,)
    _TEST_FEATURE_TYPES = (torch.Tensor, torch.Tensor, torch.Tensor)

    datasets_utils.combinations_grid(train=(True, False))

    _NAME = "liberty"

    def dataset_args(self, tmpdir, config):
        return tmpdir, self._NAME

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        # In contrast to the original data, the fake images injected here comprise only a single patch. Thus,
        # num_images == num_patches.
        num_patches = 5

        image_files = self._create_images(tmpdir, self._NAME, num_patches)
        point_ids, info_file = self._create_info_file(tmpdir / self._NAME, num_patches)
        num_matches, matches_file = self._create_matches_file(tmpdir / self._NAME, num_patches, point_ids)

        self._create_archive(tmpdir, self._NAME, *image_files, info_file, matches_file)

        return num_patches if config["train"] else num_matches

    def _create_images(self, root, name, num_images):
        # The images in the PhotoTour dataset comprises of multiple grayscale patches of 64 x 64 pixels. Thus, the
        # smallest fake image is 64 x 64 pixels and comprises a single patch.
        return datasets_utils.create_image_folder(
            root, name, lambda idx: f"patches{idx:04d}.bmp", num_images, size=(1, 64, 64)
        )

    def _create_info_file(self, root, num_images):
        point_ids = torch.randint(num_images, size=(num_images,)).tolist()

        file = root / "info.txt"
        with open(file, "w") as fh:
            fh.writelines([f"{point_id} 0\n" for point_id in point_ids])

        return point_ids, file

    def _create_matches_file(self, root, num_patches, point_ids):
        lines = [
            f"{patch_id1} {point_ids[patch_id1]} 0 {patch_id2} {point_ids[patch_id2]} 0\n"
            for patch_id1, patch_id2 in itertools.combinations(range(num_patches), 2)
        ]

        file = root / "m50_100000_100000_0.txt"
        with open(file, "w") as fh:
            fh.writelines(lines)

        return len(lines), file

    def _create_archive(self, root, name, *files):
        archive = root / f"{name}.zip"
        with zipfile.ZipFile(archive, "w") as zip:
            for file in files:
                zip.write(file, arcname=file.relative_to(root))

        return archive

    @datasets_utils.test_all_configs
    def test_feature_types(self, config):
        feature_types = self.FEATURE_TYPES
        self.FEATURE_TYPES = self._TRAIN_FEATURE_TYPES if config["train"] else self._TEST_FEATURE_TYPES
        try:
            super().test_feature_types.__wrapped__(self, config)
        finally:
            self.FEATURE_TYPES = feature_types


class Flickr8kTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Flickr8k

    FEATURE_TYPES = (PIL.Image.Image, list)

    _IMAGES_FOLDER = "images"
    _ANNOTATIONS_FILE = "captions.html"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._IMAGES_FOLDER
        ann_file = tmpdir / self._ANNOTATIONS_FILE
        return str(root), str(ann_file)

    def inject_fake_data(self, tmpdir, config):
        num_images = 3
        num_captions_per_image = 3

        tmpdir = pathlib.Path(tmpdir)

        images = self._create_images(tmpdir, self._IMAGES_FOLDER, num_images)
        self._create_annotations_file(tmpdir, self._ANNOTATIONS_FILE, images, num_captions_per_image)

        return dict(num_examples=num_images, captions=self._create_captions(num_captions_per_image))

    def _create_images(self, root, name, num_images):
        return datasets_utils.create_image_folder(root, name, self._image_file_name, num_images)

    def _image_file_name(self, idx):
        id = datasets_utils.create_random_string(10, string.digits)
        checksum = datasets_utils.create_random_string(10, string.digits, string.ascii_lowercase[:6])
        size = datasets_utils.create_random_string(1, "qwcko")
        return f"{id}_{checksum}_{size}.jpg"

    def _create_annotations_file(self, root, name, images, num_captions_per_image):
        with open(root / name, "w") as fh:
            fh.write("<table>")
            for image in (None, *images):
                self._add_image(fh, image, num_captions_per_image)
            fh.write("</table>")

    def _add_image(self, fh, image, num_captions_per_image):
        fh.write("<tr>")
        self._add_image_header(fh, image)
        fh.write("</tr><tr><td><ul>")
        self._add_image_captions(fh, num_captions_per_image)
        fh.write("</ul></td></tr>")

    def _add_image_header(self, fh, image=None):
        if image:
            url = f"http://www.flickr.com/photos/user/{image.name.split('_')[0]}/"
            data = f'<a href="{url}">{url}</a>'
        else:
            data = "Image Not Found"
        fh.write(f"<td>{data}</td>")

    def _add_image_captions(self, fh, num_captions_per_image):
        for caption in self._create_captions(num_captions_per_image):
            fh.write(f"<li>{caption}")

    def _create_captions(self, num_captions_per_image):
        return [str(idx) for idx in range(num_captions_per_image)]

    def test_captions(self):
        with self.create_dataset() as (dataset, info):
            _, captions = dataset[0]
            self.assertSequenceEqual(captions, info["captions"])


class Flickr30kTestCase(Flickr8kTestCase):
    DATASET_CLASS = datasets.Flickr30k

    FEATURE_TYPES = (PIL.Image.Image, list)

    _ANNOTATIONS_FILE = "captions.token"

    def _image_file_name(self, idx):
        return f"{idx}.jpg"

    def _create_annotations_file(self, root, name, images, num_captions_per_image):
        with open(root / name, "w") as fh:
            for image, (idx, caption) in itertools.product(
                images, enumerate(self._create_captions(num_captions_per_image))
            ):
                fh.write(f"{image.name}#{idx}\t{caption}\n")


class MNISTTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.MNIST

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    _MAGIC_DTYPES = {
        torch.uint8: 8,
        torch.int8: 9,
        torch.int16: 11,
        torch.int32: 12,
        torch.float32: 13,
        torch.float64: 14,
    }

    _IMAGES_SIZE = (28, 28)
    _IMAGES_DTYPE = torch.uint8

    _LABELS_SIZE = ()
    _LABELS_DTYPE = torch.uint8

    def inject_fake_data(self, tmpdir, config):
        raw_dir = pathlib.Path(tmpdir) / self.DATASET_CLASS.__name__ / "raw"
        os.makedirs(raw_dir, exist_ok=True)

        num_images = self._num_images(config)
        self._create_binary_file(
            raw_dir, self._images_file(config), (num_images, *self._IMAGES_SIZE), self._IMAGES_DTYPE
        )
        self._create_binary_file(
            raw_dir, self._labels_file(config), (num_images, *self._LABELS_SIZE), self._LABELS_DTYPE
        )
        return num_images

    def _num_images(self, config):
        return 2 if config["train"] else 1

    def _images_file(self, config):
        return f"{self._prefix(config)}-images-idx3-ubyte"

    def _labels_file(self, config):
        return f"{self._prefix(config)}-labels-idx1-ubyte"

    def _prefix(self, config):
        return "train" if config["train"] else "t10k"

    def _create_binary_file(self, root, filename, size, dtype):
        with open(pathlib.Path(root) / filename, "wb") as fh:
            for meta in (self._magic(dtype, len(size)), *size):
                fh.write(self._encode(meta))

            # If ever an MNIST variant is added that uses floating point data, this should be adapted.
            data = torch.randint(0, torch.iinfo(dtype).max + 1, size, dtype=dtype)
            fh.write(data.numpy().tobytes())

    def _magic(self, dtype, dims):
        return self._MAGIC_DTYPES[dtype] * 256 + dims

    def _encode(self, v):
        return torch.tensor(v, dtype=torch.int32).numpy().tobytes()[::-1]


class FashionMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.FashionMNIST


class KMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.KMNIST


class EMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.EMNIST

    DEFAULT_CONFIG = dict(split="byclass")
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("byclass", "bymerge", "balanced", "letters", "digits", "mnist"), train=(True, False)
    )

    def _prefix(self, config):
        return f"emnist-{config['split']}-{'train' if config['train'] else 'test'}"


class QMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.QMNIST

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(what=("train", "test", "test10k", "nist"))

    _LABELS_SIZE = (8,)
    _LABELS_DTYPE = torch.int32

    def _num_images(self, config):
        if config["what"] == "nist":
            return 3
        elif config["what"] == "train":
            return 2
        elif config["what"] == "test50k":
            # The split 'test50k' is defined as the last 50k images beginning at index 10000. Thus, we need to create
            # more than 10000 images for the dataset to not be empty. Since this takes significantly longer than the
            # creation of all other splits, this is excluded from the 'ADDITIONAL_CONFIGS' and is tested only once in
            # 'test_num_examples_test50k'.
            return 10001
        else:
            return 1

    def _labels_file(self, config):
        return f"{self._prefix(config)}-labels-idx2-int"

    def _prefix(self, config):
        if config["what"] == "nist":
            return "xnist"

        if config["what"] is None:
            what = "train" if config["train"] else "test"
        elif config["what"].startswith("test"):
            what = "test"
        else:
            what = config["what"]

        return f"qmnist-{what}"

    def test_num_examples_test50k(self):
        with self.create_dataset(what="test50k") as (dataset, info):
            # Since the split 'test50k' selects all images beginning from the index 10000, we subtract the number of
            # created examples by this.
            self.assertEqual(len(dataset), info["num_examples"] - 10000)


class DatasetFolderTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.DatasetFolder

    # The dataset has no fixed return type since it is defined by the loader parameter. For testing, we use a loader
    # that simply returns the path as type 'str' instead of loading anything. See the 'dataset_args()' method.
    FEATURE_TYPES = (str, int)

    _IMAGE_EXTENSIONS = ("jpg", "png")
    _VIDEO_EXTENSIONS = ("avi", "mp4")
    _EXTENSIONS = (*_IMAGE_EXTENSIONS, *_VIDEO_EXTENSIONS)

    # DatasetFolder has two mutually exclusive parameters: 'extensions' and 'is_valid_file'. One of both is required.
    # We only iterate over different 'extensions' here and handle the tests for 'is_valid_file' in the
    # 'test_is_valid_file()' method.
    DEFAULT_CONFIG = dict(extensions=_EXTENSIONS)
    ADDITIONAL_CONFIGS = (
        *datasets_utils.combinations_grid(extensions=[(ext,) for ext in _IMAGE_EXTENSIONS]),
        dict(extensions=_IMAGE_EXTENSIONS),
        *datasets_utils.combinations_grid(extensions=[(ext,) for ext in _VIDEO_EXTENSIONS]),
        dict(extensions=_VIDEO_EXTENSIONS),
    )

    def dataset_args(self, tmpdir, config):
        return tmpdir, lambda x: x

    def inject_fake_data(self, tmpdir, config):
        extensions = config["extensions"] or self._is_valid_file_to_extensions(config["is_valid_file"])

        num_examples_total = 0
        classes = []
        for ext, cls in zip(self._EXTENSIONS, string.ascii_letters):
            if ext not in extensions:
                continue

            create_example_folder = (
                datasets_utils.create_image_folder
                if ext in self._IMAGE_EXTENSIONS
                else datasets_utils.create_video_folder
            )

            num_examples = torch.randint(1, 3, size=()).item()
            create_example_folder(tmpdir, cls, lambda idx: self._file_name_fn(cls, ext, idx), num_examples)

            num_examples_total += num_examples
            classes.append(cls)

        return dict(num_examples=num_examples_total, classes=classes)

    def _file_name_fn(self, cls, ext, idx):
        return f"{cls}_{idx}.{ext}"

    def _is_valid_file_to_extensions(self, is_valid_file):
        return {ext for ext in self._EXTENSIONS if is_valid_file(f"foo.{ext}")}

    @datasets_utils.test_all_configs
    def test_is_valid_file(self, config):
        extensions = config.pop("extensions")
        # We need to explicitly pass extensions=None here or otherwise it would be filled by the value from the
        # DEFAULT_CONFIG.
        with self.create_dataset(
                config, extensions=None, is_valid_file=lambda file: pathlib.Path(file).suffix[1:] in extensions
        ) as (dataset, info):
            self.assertEqual(len(dataset), info["num_examples"])

    @datasets_utils.test_all_configs
    def test_classes(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertSequenceEqual(dataset.classes, info["classes"])


class ImageFolderTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ImageFolder

    def inject_fake_data(self, tmpdir, config):
        num_examples_total = 0
        classes = ("a", "b")
        for cls in classes:
            num_examples = torch.randint(1, 3, size=()).item()
            num_examples_total += num_examples

            datasets_utils.create_image_folder(tmpdir, cls, lambda idx: f"{cls}_{idx}.png", num_examples)

        return dict(num_examples=num_examples_total, classes=classes)

    @datasets_utils.test_all_configs
    def test_classes(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertSequenceEqual(dataset.classes, info["classes"])


class KittiTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Kitti
    FEATURE_TYPES = (PIL.Image.Image, (list, type(None)))  # test split returns None as target
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    def inject_fake_data(self, tmpdir, config):
        kitti_dir = os.path.join(tmpdir, "Kitti", "raw")
        os.makedirs(kitti_dir)

        split_to_num_examples = {
            True: 1,
            False: 2,
        }

        # We need to create all folders(training and testing).
        for is_training in (True, False):
            num_examples = split_to_num_examples[is_training]

            datasets_utils.create_image_folder(
                root=kitti_dir,
                name=os.path.join("training" if is_training else "testing", "image_2"),
                file_name_fn=lambda image_idx: f"{image_idx:06d}.png",
                num_examples=num_examples,
            )
            if is_training:
                for image_idx in range(num_examples):
                    target_file_dir = os.path.join(kitti_dir, "training", "label_2")
                    os.makedirs(target_file_dir)
                    target_file_name = os.path.join(target_file_dir, f"{image_idx:06d}.txt")
                    target_contents = "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"  # noqa
                    with open(target_file_name, "w") as target_file:
                        target_file.write(target_contents)

        return split_to_num_examples[config["train"]]


class SvhnTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SVHN
    REQUIRED_PACKAGES = ("scipy",)
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(split=("train", "test", "extra"))

    def inject_fake_data(self, tmpdir, config):
        import scipy.io as sio

        split = config["split"]
        num_examples = {
            "train": 2,
            "test": 3,
            "extra": 4,
        }.get(split)

        file = f"{split}_32x32.mat"
        images = np.zeros((32, 32, 3, num_examples), dtype=np.uint8)
        targets = np.zeros((num_examples,), dtype=np.uint8)
        sio.savemat(os.path.join(tmpdir, file), {'X': images, 'y': targets})
        return num_examples


class Places365TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Places365
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("train-standard", "train-challenge", "val"),
        small=(False, True),
    )
    _CATEGORIES = "categories_places365.txt"
    # {split: file}
    _FILE_LISTS = {
        "train-standard": "places365_train_standard.txt",
        "train-challenge": "places365_train_challenge.txt",
        "val": "places365_val.txt",
    }
    # {(split, small): folder_name}
    _IMAGES = {
        ("train-standard", False): "data_large_standard",
        ("train-challenge", False): "data_large_challenge",
        ("val", False): "val_large",
        ("train-standard", True): "data_256_standard",
        ("train-challenge", True): "data_256_challenge",
        ("val", True): "val_256",
    }
    # (class, idx)
    _CATEGORIES_CONTENT = (
        ("/a/airfield", 0),
        ("/a/apartment_building/outdoor", 8),
        ("/b/badlands", 30),
    )
    # (file, idx)
    _FILE_LIST_CONTENT = (
        ("Places365_val_00000001.png", 0),
        *((f"{category}/Places365_train_00000001.png", idx)
          for category, idx in _CATEGORIES_CONTENT),
    )

    @staticmethod
    def _make_txt(root, name, seq):
        file = os.path.join(root, name)
        with open(file, "w") as fh:
            for text, idx in seq:
                fh.write(f"{text} {idx}\n")

    @staticmethod
    def _make_categories_txt(root, name):
        Places365TestCase._make_txt(root, name, Places365TestCase._CATEGORIES_CONTENT)

    @staticmethod
    def _make_file_list_txt(root, name):
        Places365TestCase._make_txt(root, name, Places365TestCase._FILE_LIST_CONTENT)

    @staticmethod
    def _make_image(file_name, size):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        PIL.Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(file_name)

    @staticmethod
    def _make_devkit_archive(root, split):
        Places365TestCase._make_categories_txt(root, Places365TestCase._CATEGORIES)
        Places365TestCase._make_file_list_txt(root, Places365TestCase._FILE_LISTS[split])

    @staticmethod
    def _make_images_archive(root, split, small):
        folder_name = Places365TestCase._IMAGES[(split, small)]
        image_size = (256, 256) if small else (512, random.randint(512, 1024))
        files, idcs = zip(*Places365TestCase._FILE_LIST_CONTENT)
        images = [f.lstrip("/").replace("/", os.sep) for f in files]
        for image in images:
            Places365TestCase._make_image(os.path.join(root, folder_name, image), image_size)

        return [(os.path.join(root, folder_name, image), idx) for image, idx in zip(images, idcs)]

    def inject_fake_data(self, tmpdir, config):
        self._make_devkit_archive(tmpdir, config['split'])
        return len(self._make_images_archive(tmpdir, config['split'], config['small']))

    def test_classes(self):
        classes = list(map(lambda x: x[0], self._CATEGORIES_CONTENT))
        with self.create_dataset() as (dataset, _):
            self.assertEqual(dataset.classes, classes)

    def test_class_to_idx(self):
        class_to_idx = dict(self._CATEGORIES_CONTENT)
        with self.create_dataset() as (dataset, _):
            self.assertEqual(dataset.class_to_idx, class_to_idx)

    def test_images_download_preexisting(self):
        with self.assertRaises(RuntimeError):
            with self.create_dataset({'download': True}):
                pass


if __name__ == "__main__":
    unittest.main()
