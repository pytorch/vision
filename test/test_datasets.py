import bz2
import contextlib
import csv
import io
import itertools
import json
import os
import pathlib
import pickle
import random
import re
import shutil
import string
import unittest
import xml.etree.ElementTree as ET
import zipfile
from typing import Callable, Tuple, Union

import datasets_utils
import numpy as np
import PIL
import pytest
import torch
import torch.nn.functional as F
from common_utils import combinations_grid
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import v2


class STL10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.STL10
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test", "unlabeled", "train+unlabeled"))

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
                assert len(dataset) == fold + 1

    def test_unlabeled(self):
        with self.create_dataset(split="unlabeled") as (dataset, _):
            labels = [dataset[idx][1] for idx in range(len(dataset))]
            assert all(label == -1 for label in labels)

    def test_invalid_folds1(self):
        with pytest.raises(ValueError):
            with self.create_dataset(folds=10):
                pass

    def test_invalid_folds2(self):
        with pytest.raises(ValueError):
            with self.create_dataset(folds="0"):
                pass


class Caltech101TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech101
    FEATURE_TYPES = (PIL.Image.Image, (int, np.ndarray, tuple))

    ADDITIONAL_CONFIGS = combinations_grid(target_type=("category", "annotation", ["category", "annotation"]))
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
        assert (
            actual == expected
        ), "The number of the returned combined targets does not match the the number targets if requested "
        f"individually: {actual} != {expected}",

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                assert (
                    actual is expected
                ), "Type of the combined target does not match the type of the corresponding individual target: "
                f"{actual} is not {expected}",

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(target_type="category", transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class Caltech256TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech256

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / "caltech256" / "256_ObjectCategories"

        categories = ((1, "ak47"), (2, "american-flag"), (3, "backpack"))
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
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))

    def inject_fake_data(self, tmpdir, config):
        widerface_dir = pathlib.Path(tmpdir) / "widerface"
        annotations_dir = widerface_dir / "wider_face_split"
        os.makedirs(annotations_dir)

        split_to_idx = split_to_num_examples = {
            "train": 1,
            "val": 2,
            "test": 3,
        }

        # We need to create all folders regardless of the split in config
        for split in ("train", "val", "test"):
            split_idx = split_to_idx[split]
            num_examples = split_to_num_examples[split]

            datasets_utils.create_image_folder(
                root=tmpdir,
                name=widerface_dir / f"WIDER_{split}" / "images" / "0--Parade",
                file_name_fn=lambda image_idx: f"0_Parade_marchingband_1_{split_idx + image_idx}.jpg",
                num_examples=num_examples,
            )

            annotation_file_name = {
                "train": annotations_dir / "wider_face_train_bbx_gt.txt",
                "val": annotations_dir / "wider_face_val_bbx_gt.txt",
                "test": annotations_dir / "wider_face_test_filelist.txt",
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

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class CityScapesTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Cityscapes
    TARGET_TYPES = (
        "instance",
        "semantic",
        "polygon",
        "color",
    )
    ADDITIONAL_CONFIGS = (
        *combinations_grid(mode=("fine",), split=("train", "test", "val"), target_type=TARGET_TYPES),
        *combinations_grid(
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
        for split in ["test", "train_extra", "train", "val"]:
            for city in cities:
                datasets_utils.create_image_folder(
                    root=tmpdir / "leftImg8bit" / split,
                    name=city,
                    file_name_fn=lambda _: f"{city}_000000_000000_leftImg8bit.png",
                    num_examples=1,
                )

        info = {"num_examples": len(cities)}
        if config["target_type"] == "polygon":
            info["expected_polygon_target"] = polygon_target
        return info

    def test_combined_targets(self):
        target_types = ["semantic", "polygon", "color"]

        with self.create_dataset(target_type=target_types) as (dataset, _):
            output = dataset[0]
            assert isinstance(output, tuple)
            assert len(output) == 2
            assert isinstance(output[0], PIL.Image.Image)
            assert isinstance(output[1], tuple)
            assert len(output[1]) == 3
            assert isinstance(output[1][0], PIL.Image.Image)  # semantic
            assert isinstance(output[1][1], dict)  # polygon
            assert isinstance(output[1][2], PIL.Image.Image)  # color

    def test_feature_types_target_color(self):
        with self.create_dataset(target_type="color") as (dataset, _):
            color_img, color_target = dataset[0]
            assert isinstance(color_img, PIL.Image.Image)
            assert np.array(color_target).shape[2] == 4

    def test_feature_types_target_polygon(self):
        with self.create_dataset(target_type="polygon") as (dataset, info):
            polygon_img, polygon_target = dataset[0]
            assert isinstance(polygon_img, PIL.Image.Image)
            (polygon_target, info["expected_polygon_target"])

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        for target_type in ["instance", "semantic", ["instance", "semantic"]]:
            with self.create_dataset(target_type=target_type, transform=v2.Resize(size=expected_size)) as (dataset, _):
                datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class ImageNetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ImageNet
    REQUIRED_PACKAGES = ("scipy",)
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val"))

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        wnid = "n01234567"
        if config["split"] == "train":
            num_examples = 3
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / "train" / wnid / wnid,
                file_name_fn=lambda image_idx: f"{wnid}_{image_idx}.JPEG",
                num_examples=num_examples,
            )
        else:
            num_examples = 1
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / "val" / wnid,
                file_name_fn=lambda image_ifx: "ILSVRC2012_val_0000000{image_idx}.JPEG",
                num_examples=num_examples,
            )

        wnid_to_classes = {wnid: [1]}
        torch.save((wnid_to_classes, None), tmpdir / "meta.bin")
        return num_examples

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class CIFAR10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CIFAR10
    ADDITIONAL_CONFIGS = combinations_grid(train=(True, False))

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
        np_rng = np.random.RandomState(0)
        data = datasets_utils.create_image_or_video_tensor((num_images, 32 * 32 * 3))
        labels = np_rng.randint(0, self._VERSION_CONFIG["num_categories"], size=num_images).tolist()
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
            assert actual == expected


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

    ADDITIONAL_CONFIGS = combinations_grid(
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

        num_samples = num_images_per_split.get(config["split"], 0) if isinstance(config["split"], str) else 0
        return dict(num_examples=num_samples, attr_names=attr_names)

    def _create_split_txt(self, root):
        num_images_per_split = dict(train=4, valid=3, test=2)

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
        assert (
            actual == expected
        ), "The number of the returned combined targets does not match the the number targets if requested "
        f"individually: {actual} != {expected}",

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                assert (
                    actual is expected
                ), "Type of the combined target does not match the type of the corresponding individual target: "
                f"{actual} is not {expected}",

    def test_no_target(self):
        with self.create_dataset(target_type=[]) as (dataset, _):
            _, target = dataset[0]

        assert target is None

    def test_attr_names(self):
        with self.create_dataset() as (dataset, info):
            assert tuple(dataset.attr_names) == info["attr_names"]

    def test_images_names_split(self):
        with self.create_dataset(split="all") as (dataset, _):
            all_imgs_names = set(dataset.filename)

        merged_imgs_names = set()
        for split in ["train", "valid", "test"]:
            with self.create_dataset(split=split) as (dataset, _):
                merged_imgs_names.update(dataset.filename)

        assert merged_imgs_names == all_imgs_names

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        for target_type in ["identity", "bbox", ["identity", "bbox"]]:
            with self.create_dataset(target_type=target_type, transform=v2.Resize(size=expected_size)) as (dataset, _):
                datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)

    def test_invalid_split_list(self):
        with pytest.raises(ValueError, match="Expected type str for argument split, but got type <class 'list'>."):
            with self.create_dataset(split=[1]):
                pass

    def test_invalid_split_int(self):
        with pytest.raises(ValueError, match="Expected type str for argument split, but got type <class 'int'>."):
            with self.create_dataset(split=1):
                pass

    def test_invalid_split_value(self):
        with pytest.raises(
            ValueError,
            match="Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}.".format(
                value="invalid",
                arg="split",
                valid_values=("train", "valid", "test", "all"),
            ),
        ):
            with self.create_dataset(split="invalid"):
                pass


class VOCSegmentationTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.VOCSegmentation
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image)

    ADDITIONAL_CONFIGS = (
        *combinations_grid(year=[f"20{year:02d}" for year in range(7, 13)], image_set=("train", "val", "trainval")),
        dict(year="2007", image_set="test"),
    )

    def inject_fake_data(self, tmpdir, config):
        year, is_test_set = config["year"], config["image_set"] == "test"
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
        num_images_per_image_set = {image_set: len(idcs_) for image_set, idcs_ in idcs.items()}
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

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class VOCDetectionTestCase(VOCSegmentationTestCase):
    DATASET_CLASS = datasets.VOCDetection
    FEATURE_TYPES = (PIL.Image.Image, dict)

    def test_annotations(self):
        with self.create_dataset() as (dataset, info):
            _, target = dataset[0]

            assert "annotation" in target
            annotation = target["annotation"]

            assert "object" in annotation
            objects = annotation["object"]

            assert len(objects) == 1
            object = objects[0]

            assert object == info["annotation"]

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


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

        segmentation_kind = config.pop("segmentation_kind", "list")
        info = self._create_annotation_file(
            annotation_folder,
            self._ANNOTATIONS_FILE,
            file_names,
            num_annotations_per_image,
            segmentation_kind=segmentation_kind,
        )

        info["num_examples"] = num_images
        return info

    def _create_annotation_file(self, root, name, file_names, num_annotations_per_image, segmentation_kind="list"):
        image_ids = [int(file_name.stem) for file_name in file_names]
        images = [dict(file_name=str(file_name), id=id) for file_name, id in zip(file_names, image_ids)]

        annotations, info = self._create_annotations(image_ids, num_annotations_per_image, segmentation_kind)
        self._create_json(root, name, dict(images=images, annotations=annotations))

        return info

    def _create_annotations(self, image_ids, num_annotations_per_image, segmentation_kind="list"):
        annotations = []
        annotion_id = 0

        for image_id in itertools.islice(itertools.cycle(image_ids), len(image_ids) * num_annotations_per_image):
            segmentation = {
                "list": [torch.rand(8).tolist()],
                "rle": {"size": [10, 10], "counts": [1]},
                "rle_encoded": {"size": [2400, 2400], "counts": "PQRQ2[1\\Y2f0gNVNRhMg2"},
                "bad": 123,
            }[segmentation_kind]

            annotations.append(
                dict(
                    image_id=image_id,
                    id=annotion_id,
                    bbox=torch.rand(4).tolist(),
                    segmentation=segmentation,
                    category_id=int(torch.randint(91, ())),
                    area=float(torch.rand(1)),
                    iscrowd=int(torch.randint(2, size=(1,))),
                )
            )
            annotion_id += 1
        return annotations, dict()

    def _create_json(self, root, name, content):
        file = pathlib.Path(root) / name
        with open(file, "w") as fh:
            json.dump(content, fh)
        return file

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)

    def test_slice_error(self):
        with self.create_dataset() as (dataset, _):
            with pytest.raises(ValueError, match="Index must be of type integer"):
                dataset[:2]

    def test_segmentation_kind(self):
        if isinstance(self, CocoCaptionsTestCase):
            return

        for segmentation_kind in ("list", "rle", "rle_encoded"):
            config = {"segmentation_kind": segmentation_kind}
            with self.create_dataset(config) as (dataset, _):
                dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys="all")
                list(dataset)

        config = {"segmentation_kind": "bad"}
        with self.create_dataset(config) as (dataset, _):
            dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys="all")
            with pytest.raises(ValueError, match="COCO segmentation expected to be a dict or a list"):
                list(dataset)


class CocoCaptionsTestCase(CocoDetectionTestCase):
    DATASET_CLASS = datasets.CocoCaptions

    def _create_annotations(self, image_ids, num_annotations_per_image, segmentation_kind="list"):
        captions = [str(idx) for idx in range(num_annotations_per_image)]
        annotations = combinations_grid(image_id=image_ids, caption=captions)
        for id, annotation in enumerate(annotations):
            annotation["id"] = id
        return annotations, dict(captions=captions)

    def test_captions(self):
        with self.create_dataset() as (dataset, info):
            _, captions = dataset[0]
            assert tuple(captions) == tuple(info["captions"])

    def test_transforms_v2_wrapper_spawn(self):
        # We need to define this method, because otherwise the test from the super class will
        # be run
        pytest.skip("CocoCaptions is currently not supported by the v2 wrapper.")


class UCF101TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.UCF101

    ADDITIONAL_CONFIGS = combinations_grid(fold=(1, 2, 3), train=(True, False))

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
            fh.writelines(f"{str(file).replace(os.sep, '/')}\n" for file in sorted(video_files))


class LSUNTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.LSUN

    REQUIRED_PACKAGES = ("lmdb",)
    ADDITIONAL_CONFIGS = combinations_grid(classes=("train", "test", "val", ["bedroom_train", "church_outdoor_train"]))

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
    def create_dataset(self, *args, **kwargs):
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
        with pytest.raises(datasets_utils.lazy_importer.lmdb.Error):
            super().test_not_found_or_corrupted()


class KineticsTestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.Kinetics
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val"), num_classes=("400", "600", "700"))

    def inject_fake_data(self, tmpdir, config):
        classes = ("Abseiling", "Zumba")
        num_videos_per_class = 2
        tmpdir = pathlib.Path(tmpdir) / config["split"]
        digits = string.ascii_letters + string.digits + "-_"
        for cls in classes:
            datasets_utils.create_video_folder(
                tmpdir,
                cls,
                lambda _: f"{datasets_utils.create_random_string(11, digits)}.mp4",
                num_videos_per_class,
            )
        return num_videos_per_class * len(classes)

    @pytest.mark.xfail(reason="FIXME")
    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(output_format="TCHW", transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class HMDB51TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.HMDB51

    ADDITIONAL_CONFIGS = combinations_grid(fold=(1, 2, 3), train=(True, False))

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

    ADDITIONAL_CONFIGS = combinations_grid(background=(True, False))
    SUPPORT_TV_IMAGE_DECODE = True

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

    SUPPORT_TV_IMAGE_DECODE = True

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

    ADDITIONAL_CONFIGS = combinations_grid(train=(True, False))

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

    ADDITIONAL_CONFIGS = combinations_grid(
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
        num_images_per_split = {split: len(idcs) for split, idcs in splits.items()}
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

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(mode="segmentation", transforms=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


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

    combinations_grid(train=(True, False))

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

    SUPPORT_TV_IMAGE_DECODE = True

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
            assert len(captions) == len(info["captions"])
            assert all([a == b for a, b in zip(captions, info["captions"])])


class Flickr30kTestCase(Flickr8kTestCase):
    DATASET_CLASS = datasets.Flickr30k

    FEATURE_TYPES = (PIL.Image.Image, list)

    _ANNOTATIONS_FILE = "captions.token"

    SUPPORT_TV_IMAGE_DECODE = True

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

    ADDITIONAL_CONFIGS = combinations_grid(train=(True, False))

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
    ADDITIONAL_CONFIGS = combinations_grid(
        split=("byclass", "bymerge", "balanced", "letters", "digits", "mnist"), train=(True, False)
    )

    def _prefix(self, config):
        return f"emnist-{config['split']}-{'train' if config['train'] else 'test'}"


class QMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.QMNIST

    ADDITIONAL_CONFIGS = combinations_grid(what=("train", "test", "test10k", "nist"))

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
            assert len(dataset) == info["num_examples"] - 10000


class MovingMNISTTestCase(datasets_utils.DatasetTestCase):
    DATASET_CLASS = datasets.MovingMNIST
    FEATURE_TYPES = (torch.Tensor,)

    ADDITIONAL_CONFIGS = combinations_grid(split=(None, "train", "test"), split_ratio=(10, 1, 19))

    _NUM_FRAMES = 20

    def inject_fake_data(self, tmpdir, config):
        base_folder = os.path.join(tmpdir, self.DATASET_CLASS.__name__)
        os.makedirs(base_folder, exist_ok=True)
        num_samples = 5
        data = np.concatenate(
            [
                np.zeros((config["split_ratio"], num_samples, 64, 64)),
                np.ones((self._NUM_FRAMES - config["split_ratio"], num_samples, 64, 64)),
            ]
        )
        np.save(os.path.join(base_folder, "mnist_test_seq.npy"), data)
        return num_samples

    @datasets_utils.test_all_configs
    def test_split(self, config):
        with self.create_dataset(config) as (dataset, _):
            if config["split"] == "train":
                assert (dataset.data == 0).all()
            elif config["split"] == "test":
                assert (dataset.data == 1).all()
            else:
                assert dataset.data.size()[1] == self._NUM_FRAMES


class DatasetFolderTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.DatasetFolder

    _EXTENSIONS = ("jpg", "png")

    # DatasetFolder has two mutually exclusive parameters: 'extensions' and 'is_valid_file'. One of both is required.
    # We only iterate over different 'extensions' here and handle the tests for 'is_valid_file' in the
    # 'test_is_valid_file()' method.
    DEFAULT_CONFIG = dict(extensions=_EXTENSIONS)
    ADDITIONAL_CONFIGS = combinations_grid(extensions=[(ext,) for ext in _EXTENSIONS])

    def dataset_args(self, tmpdir, config):
        return tmpdir, datasets.folder.pil_loader

    def inject_fake_data(self, tmpdir, config):
        extensions = config["extensions"] or self._is_valid_file_to_extensions(config["is_valid_file"])

        num_examples_total = 0
        classes = []
        for ext, cls in zip(self._EXTENSIONS, string.ascii_letters):
            if ext not in extensions:
                continue

            num_examples = torch.randint(1, 3, size=()).item()
            datasets_utils.create_image_folder(tmpdir, cls, lambda idx: self._file_name_fn(cls, ext, idx), num_examples)

            num_examples_total += num_examples
            classes.append(cls)

        if config.pop("make_empty_class", False):
            os.makedirs(pathlib.Path(tmpdir) / "empty_class")
            classes.append("empty_class")

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
            assert len(dataset) == info["num_examples"]

    @datasets_utils.test_all_configs
    def test_classes(self, config):
        with self.create_dataset(config) as (dataset, info):
            assert len(dataset.classes) == len(info["classes"])
            assert all([a == b for a, b in zip(dataset.classes, info["classes"])])

    def test_allow_empty(self):
        config = {
            "extensions": self._EXTENSIONS,
            "make_empty_class": True,
        }

        config["allow_empty"] = True
        with self.create_dataset(config) as (dataset, info):
            assert "empty_class" in dataset.classes
            assert len(dataset.classes) == len(info["classes"])
            assert all([a == b for a, b in zip(dataset.classes, info["classes"])])

        config["allow_empty"] = False
        with pytest.raises(FileNotFoundError, match="Found no valid file"):
            with self.create_dataset(config) as (dataset, info):
                pass


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
            assert len(dataset.classes) == len(info["classes"])
            assert all([a == b for a, b in zip(dataset.classes, info["classes"])])


class KittiTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Kitti
    FEATURE_TYPES = (PIL.Image.Image, (list, type(None)))  # test split returns None as target
    ADDITIONAL_CONFIGS = combinations_grid(train=(True, False))

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

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class SvhnTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SVHN
    REQUIRED_PACKAGES = ("scipy",)
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test", "extra"))

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
        sio.savemat(os.path.join(tmpdir, file), {"X": images, "y": targets})
        return num_examples


class Places365TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Places365
    ADDITIONAL_CONFIGS = combinations_grid(
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
        *((f"{category}/Places365_train_00000001.png", idx) for category, idx in _CATEGORIES_CONTENT),
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
        self._make_devkit_archive(tmpdir, config["split"])
        return len(self._make_images_archive(tmpdir, config["split"], config["small"]))

    def test_classes(self):
        classes = list(map(lambda x: x[0], self._CATEGORIES_CONTENT))
        with self.create_dataset() as (dataset, _):
            assert dataset.classes == classes

    def test_class_to_idx(self):
        class_to_idx = dict(self._CATEGORIES_CONTENT)
        with self.create_dataset() as (dataset, _):
            assert dataset.class_to_idx == class_to_idx


class INaturalistTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.INaturalist
    FEATURE_TYPES = (PIL.Image.Image, (int, tuple))

    ADDITIONAL_CONFIGS = combinations_grid(
        target_type=("kingdom", "full", "genus", ["kingdom", "phylum", "class", "order", "family", "genus", "full"]),
        version=("2021_train",),
    )
    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        categories = [
            "00000_Akingdom_0phylum_Aclass_Aorder_Afamily_Agenus_Aspecies",
            "00001_Akingdom_1phylum_Aclass_Border_Afamily_Bgenus_Aspecies",
            "00002_Akingdom_2phylum_Cclass_Corder_Cfamily_Cgenus_Cspecies",
        ]

        num_images_per_category = 3
        for category in categories:
            datasets_utils.create_image_folder(
                root=os.path.join(tmpdir, config["version"]),
                name=category,
                file_name_fn=lambda idx: f"image_{idx + 1:04d}.jpg",
                num_examples=num_images_per_category,
            )

        return num_images_per_category * len(categories)

    def test_targets(self):
        target_types = ["kingdom", "phylum", "class", "order", "family", "genus", "full"]

        with self.create_dataset(target_type=target_types, version="2021_valid") as (dataset, _):
            items = [d[1] for d in dataset]
            for i, item in enumerate(items):
                assert dataset.category_name("kingdom", item[0]) == "Akingdom"
                assert dataset.category_name("phylum", item[1]) == f"{i // 3}phylum"
                assert item[6] == i // 3


class LFWPeopleTestCase(datasets_utils.DatasetTestCase):
    DATASET_CLASS = datasets.LFWPeople
    FEATURE_TYPES = (PIL.Image.Image, int)
    ADDITIONAL_CONFIGS = combinations_grid(
        split=("10fold", "train", "test"), image_set=("original", "funneled", "deepfunneled")
    )
    _IMAGES_DIR = {"original": "lfw", "funneled": "lfw_funneled", "deepfunneled": "lfw-deepfunneled"}
    _file_id = {"10fold": "", "train": "DevTrain", "test": "DevTest"}

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / "lfw-py"
        os.makedirs(tmpdir, exist_ok=True)
        return dict(
            num_examples=self._create_images_dir(tmpdir, self._IMAGES_DIR[config["image_set"]], config["split"]),
            split=config["split"],
        )

    def _create_images_dir(self, root, idir, split):
        idir = os.path.join(root, idir)
        os.makedirs(idir, exist_ok=True)
        n, flines = (10, ["10\n"]) if split == "10fold" else (1, [])
        num_examples = 0
        names = []
        for _ in range(n):
            num_people = random.randint(2, 5)
            flines.append(f"{num_people}\n")
            for i in range(num_people):
                name = self._create_random_id()
                no = random.randint(1, 10)
                flines.append(f"{name}\t{no}\n")
                names.append(f"{name}\t{no}\n")
                datasets_utils.create_image_folder(idir, name, lambda n: f"{name}_{n+1:04d}.jpg", no, 250)
                num_examples += no
        with open(pathlib.Path(root) / f"people{self._file_id[split]}.txt", "w") as f:
            f.writelines(flines)
        with open(pathlib.Path(root) / "lfw-names.txt", "w") as f:
            f.writelines(sorted(names))

        return num_examples

    def _create_random_id(self):
        part1 = datasets_utils.create_random_string(random.randint(5, 7))
        part2 = datasets_utils.create_random_string(random.randint(4, 7))
        return f"{part1}_{part2}"

    def test_tv_decode_image_support(self):
        if not self.SUPPORT_TV_IMAGE_DECODE:
            pytest.skip(f"{self.DATASET_CLASS.__name__} does not support torchvision.io.decode_image.")

        with self.create_dataset(
            config=dict(
                loader=decode_image,
            )
        ) as (dataset, _):
            image = dataset[0][0]
            assert isinstance(image, torch.Tensor)


class LFWPairsTestCase(LFWPeopleTestCase):
    DATASET_CLASS = datasets.LFWPairs
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, int)

    def _create_images_dir(self, root, idir, split):
        idir = os.path.join(root, idir)
        os.makedirs(idir, exist_ok=True)
        num_pairs = 7  # effectively 7*2*n = 14*n
        n, self.flines = (10, [f"10\t{num_pairs}"]) if split == "10fold" else (1, [str(num_pairs)])
        for _ in range(n):
            self._inject_pairs(idir, num_pairs, True)
            self._inject_pairs(idir, num_pairs, False)
            with open(pathlib.Path(root) / f"pairs{self._file_id[split]}.txt", "w") as f:
                f.writelines(self.flines)

        return num_pairs * 2 * n

    def _inject_pairs(self, root, num_pairs, same):
        for i in range(num_pairs):
            name1 = self._create_random_id()
            name2 = name1 if same else self._create_random_id()
            no1, no2 = random.randint(1, 100), random.randint(1, 100)
            if same:
                self.flines.append(f"\n{name1}\t{no1}\t{no2}")
            else:
                self.flines.append(f"\n{name1}\t{no1}\t{name2}\t{no2}")

            datasets_utils.create_image_folder(root, name1, lambda _: f"{name1}_{no1:04d}.jpg", 1, 250)
            datasets_utils.create_image_folder(root, name2, lambda _: f"{name2}_{no2:04d}.jpg", 1, 250)


class SintelTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Sintel
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"), pass_name=("clean", "final", "both"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))

    FLOW_H, FLOW_W = 3, 4

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "Sintel"

        num_images_per_scene = 3 if config["split"] == "train" else 4
        num_scenes = 2

        for split_dir in ("training", "test"):
            for pass_name in ("clean", "final"):
                image_root = root / split_dir / pass_name

                for scene_id in range(num_scenes):
                    scene_dir = image_root / f"scene_{scene_id}"
                    datasets_utils.create_image_folder(
                        image_root,
                        name=str(scene_dir),
                        file_name_fn=lambda image_idx: f"frame_000{image_idx}.png",
                        num_examples=num_images_per_scene,
                    )

        flow_root = root / "training" / "flow"
        for scene_id in range(num_scenes):
            scene_dir = flow_root / f"scene_{scene_id}"
            os.makedirs(scene_dir)
            for i in range(num_images_per_scene - 1):
                file_name = str(scene_dir / f"frame_000{i}.flo")
                datasets_utils.make_fake_flo_file(h=self.FLOW_H, w=self.FLOW_W, file_name=file_name)

        # with e.g. num_images_per_scene = 3, for a single scene with have 3 images
        # which are frame_0000, frame_0001 and frame_0002
        # They will be consecutively paired as (frame_0000, frame_0001), (frame_0001, frame_0002),
        # that is 3 - 1 = 2 examples. Hence the formula below
        num_passes = 2 if config["pass_name"] == "both" else 1
        num_examples = (num_images_per_scene - 1) * num_scenes * num_passes
        return num_examples

    def test_flow(self):
        # Make sure flow exists for train split, and make sure there are as many flow values as (pairs of) images
        h, w = self.FLOW_H, self.FLOW_W
        expected_flow = np.arange(2 * h * w).reshape(h, w, 2).transpose(2, 0, 1)
        with self.create_dataset(split="train") as (dataset, _):
            assert dataset._flow_list and len(dataset._flow_list) == len(dataset._image_list)
            for _, _, flow in dataset:
                assert flow.shape == (2, h, w)
                np.testing.assert_allclose(flow, expected_flow)

        # Make sure flow is always None for test split
        with self.create_dataset(split="test") as (dataset, _):
            assert dataset._image_list and not dataset._flow_list
            for _, _, flow in dataset:
                assert flow is None

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass

        with pytest.raises(ValueError, match="Unknown value 'bad' for argument pass_name"):
            with self.create_dataset(pass_name="bad"):
                pass


class KittiFlowTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.KittiFlow
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "KittiFlow"

        num_examples = 2 if config["split"] == "train" else 3
        for split_dir in ("training", "testing"):

            datasets_utils.create_image_folder(
                root / split_dir,
                name="image_2",
                file_name_fn=lambda image_idx: f"{image_idx}_10.png",
                num_examples=num_examples,
            )
            datasets_utils.create_image_folder(
                root / split_dir,
                name="image_2",
                file_name_fn=lambda image_idx: f"{image_idx}_11.png",
                num_examples=num_examples,
            )

        # For kitti the ground truth flows are encoded as 16-bits pngs.
        # create_image_folder() will actually create 8-bits pngs, but it doesn't
        # matter much: the flow reader will still be able to read the files, it
        # will just be garbage flow value - but we don't care about that here.
        datasets_utils.create_image_folder(
            root / "training",
            name="flow_occ",
            file_name_fn=lambda image_idx: f"{image_idx}_10.png",
            num_examples=num_examples,
        )

        return num_examples

    def test_flow_and_valid(self):
        # Make sure flow exists for train split, and make sure there are as many flow values as (pairs of) images
        # Also assert flow and valid are of the expected shape
        with self.create_dataset(split="train") as (dataset, _):
            assert dataset._flow_list and len(dataset._flow_list) == len(dataset._image_list)
            for _, _, flow, valid in dataset:
                two, h, w = flow.shape
                assert two == 2
                assert valid.shape == (h, w)

        # Make sure flow and valid are always None for test split
        with self.create_dataset(split="test") as (dataset, _):
            assert dataset._image_list and not dataset._flow_list
            for _, _, flow, valid in dataset:
                assert flow is None
                assert valid is None

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass


class FlyingChairsTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FlyingChairs
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))

    FLOW_H, FLOW_W = 3, 4

    def _make_split_file(self, root, num_examples):
        # We create a fake split file here, but users are asked to download the real one from the authors website
        split_ids = [1] * num_examples["train"] + [2] * num_examples["val"]
        random.shuffle(split_ids)
        with open(str(root / "FlyingChairs_train_val.txt"), "w+") as split_file:
            for split_id in split_ids:
                split_file.write(f"{split_id}\n")

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "FlyingChairs"

        num_examples = {"train": 5, "val": 3}
        num_examples_total = sum(num_examples.values())

        datasets_utils.create_image_folder(  # img1
            root,
            name="data",
            file_name_fn=lambda image_idx: f"00{image_idx}_img1.ppm",
            num_examples=num_examples_total,
        )
        datasets_utils.create_image_folder(  # img2
            root,
            name="data",
            file_name_fn=lambda image_idx: f"00{image_idx}_img2.ppm",
            num_examples=num_examples_total,
        )
        for i in range(num_examples_total):
            file_name = str(root / "data" / f"00{i}_flow.flo")
            datasets_utils.make_fake_flo_file(h=self.FLOW_H, w=self.FLOW_W, file_name=file_name)

        self._make_split_file(root, num_examples)

        return num_examples[config["split"]]

    @datasets_utils.test_all_configs
    def test_flow(self, config):
        # Make sure flow always exists, and make sure there are as many flow values as (pairs of) images
        # Also make sure the flow is properly decoded

        h, w = self.FLOW_H, self.FLOW_W
        expected_flow = np.arange(2 * h * w).reshape(h, w, 2).transpose(2, 0, 1)
        with self.create_dataset(config=config) as (dataset, _):
            assert dataset._flow_list and len(dataset._flow_list) == len(dataset._image_list)
            for _, _, flow in dataset:
                assert flow.shape == (2, h, w)
                np.testing.assert_allclose(flow, expected_flow)


class FlyingThings3DTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FlyingThings3D
    ADDITIONAL_CONFIGS = combinations_grid(
        split=("train", "test"), pass_name=("clean", "final", "both"), camera=("left", "right", "both")
    )
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))

    FLOW_H, FLOW_W = 3, 4

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "FlyingThings3D"

        num_images_per_camera = 3 if config["split"] == "train" else 4
        passes = ("frames_cleanpass", "frames_finalpass")
        splits = ("TRAIN", "TEST")
        letters = ("A", "B", "C")
        subfolders = ("0000", "0001")
        cameras = ("left", "right")
        for pass_name, split, letter, subfolder, camera in itertools.product(
            passes, splits, letters, subfolders, cameras
        ):
            current_folder = root / pass_name / split / letter / subfolder
            datasets_utils.create_image_folder(
                current_folder,
                name=camera,
                file_name_fn=lambda image_idx: f"00{image_idx}.png",
                num_examples=num_images_per_camera,
            )

        directions = ("into_future", "into_past")
        for split, letter, subfolder, direction, camera in itertools.product(
            splits, letters, subfolders, directions, cameras
        ):
            current_folder = root / "optical_flow" / split / letter / subfolder / direction / camera
            os.makedirs(str(current_folder), exist_ok=True)
            for i in range(num_images_per_camera):
                datasets_utils.make_fake_pfm_file(self.FLOW_H, self.FLOW_W, file_name=str(current_folder / f"{i}.pfm"))

        num_cameras = 2 if config["camera"] == "both" else 1
        num_passes = 2 if config["pass_name"] == "both" else 1
        num_examples = (
            (num_images_per_camera - 1) * num_cameras * len(subfolders) * len(letters) * len(splits) * num_passes
        )
        return num_examples

    @datasets_utils.test_all_configs
    def test_flow(self, config):
        h, w = self.FLOW_H, self.FLOW_W
        expected_flow = np.arange(3 * h * w).reshape(h, w, 3).transpose(2, 0, 1)
        expected_flow = np.flip(expected_flow, axis=1)
        expected_flow = expected_flow[:2, :, :]

        with self.create_dataset(config=config) as (dataset, _):
            assert dataset._flow_list and len(dataset._flow_list) == len(dataset._image_list)
            for _, _, flow in dataset:
                assert flow.shape == (2, self.FLOW_H, self.FLOW_W)
                np.testing.assert_allclose(flow, expected_flow)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass

        with pytest.raises(ValueError, match="Unknown value 'bad' for argument pass_name"):
            with self.create_dataset(pass_name="bad"):
                pass

        with pytest.raises(ValueError, match="Unknown value 'bad' for argument camera"):
            with self.create_dataset(camera="bad"):
                pass


class HD1KTestCase(KittiFlowTestCase):
    DATASET_CLASS = datasets.HD1K

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "hd1k"

        num_sequences = 4 if config["split"] == "train" else 3
        num_examples_per_train_sequence = 3

        for seq_idx in range(num_sequences):
            # Training data
            datasets_utils.create_image_folder(
                root / "hd1k_input",
                name="image_2",
                file_name_fn=lambda image_idx: f"{seq_idx:06d}_{image_idx}.png",
                num_examples=num_examples_per_train_sequence,
            )
            datasets_utils.create_image_folder(
                root / "hd1k_flow_gt",
                name="flow_occ",
                file_name_fn=lambda image_idx: f"{seq_idx:06d}_{image_idx}.png",
                num_examples=num_examples_per_train_sequence,
            )

            # Test data
            datasets_utils.create_image_folder(
                root / "hd1k_challenge",
                name="image_2",
                file_name_fn=lambda _: f"{seq_idx:06d}_10.png",
                num_examples=1,
            )
            datasets_utils.create_image_folder(
                root / "hd1k_challenge",
                name="image_2",
                file_name_fn=lambda _: f"{seq_idx:06d}_11.png",
                num_examples=1,
            )

        num_examples_per_sequence = num_examples_per_train_sequence if config["split"] == "train" else 2
        return num_sequences * (num_examples_per_sequence - 1)


class EuroSATTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.EuroSAT
    FEATURE_TYPES = (PIL.Image.Image, int)
    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        data_folder = os.path.join(tmpdir, "eurosat", "2750")
        os.makedirs(data_folder)

        num_examples_per_class = 3
        classes = ("AnnualCrop", "Forest")
        for cls in classes:
            datasets_utils.create_image_folder(
                root=data_folder,
                name=cls,
                file_name_fn=lambda idx: f"{cls}_{idx}.jpg",
                num_examples=num_examples_per_class,
            )

        return len(classes) * num_examples_per_class


class Food101TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Food101
    FEATURE_TYPES = (PIL.Image.Image, int)

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        root_folder = pathlib.Path(tmpdir) / "food-101"
        image_folder = root_folder / "images"
        meta_folder = root_folder / "meta"

        image_folder.mkdir(parents=True)
        meta_folder.mkdir()

        num_images_per_class = 5

        metadata = {}
        n_samples_per_class = 3 if config["split"] == "train" else 2
        sampled_classes = ("apple_pie", "crab_cakes", "gyoza")
        for cls in sampled_classes:
            im_fnames = datasets_utils.create_image_folder(
                image_folder,
                cls,
                file_name_fn=lambda idx: f"{idx}.jpg",
                num_examples=num_images_per_class,
            )
            metadata[cls] = [
                "/".join(fname.relative_to(image_folder).with_suffix("").parts)
                for fname in random.choices(im_fnames, k=n_samples_per_class)
            ]

        with open(meta_folder / f"{config['split']}.json", "w") as file:
            file.write(json.dumps(metadata))

        return len(sampled_classes * n_samples_per_class)


class FGVCAircraftTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FGVCAircraft
    ADDITIONAL_CONFIGS = combinations_grid(
        split=("train", "val", "trainval", "test"), annotation_level=("variant", "family", "manufacturer")
    )
    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        split = config["split"]
        annotation_level = config["annotation_level"]
        annotation_level_to_file = {
            "variant": "variants.txt",
            "family": "families.txt",
            "manufacturer": "manufacturers.txt",
        }

        root_folder = pathlib.Path(tmpdir) / "fgvc-aircraft-2013b"
        data_folder = root_folder / "data"

        classes = ["707-320", "Hawk T1", "Tornado"]
        num_images_per_class = 5

        datasets_utils.create_image_folder(
            data_folder,
            "images",
            file_name_fn=lambda idx: f"{idx}.jpg",
            num_examples=num_images_per_class * len(classes),
        )

        annotation_file = data_folder / annotation_level_to_file[annotation_level]
        with open(annotation_file, "w") as file:
            file.write("\n".join(classes))

        num_samples_per_class = 4 if split == "trainval" else 2
        images_classes = []
        for i in range(len(classes)):
            images_classes.extend(
                [
                    f"{idx} {classes[i]}"
                    for idx in random.sample(
                        range(i * num_images_per_class, (i + 1) * num_images_per_class), num_samples_per_class
                    )
                ]
            )

        images_annotation_file = data_folder / f"images_{annotation_level}_{split}.txt"
        with open(images_annotation_file, "w") as file:
            file.write("\n".join(images_classes))

        return len(classes * num_samples_per_class)


class SUN397TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SUN397

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        data_dir = pathlib.Path(tmpdir) / "SUN397"
        data_dir.mkdir()

        num_images_per_class = 5
        sampled_classes = ("abbey", "airplane_cabin", "airport_terminal")
        im_paths = []

        for cls in sampled_classes:
            image_folder = data_dir / cls[0]
            im_paths.extend(
                datasets_utils.create_image_folder(
                    image_folder,
                    image_folder / cls,
                    file_name_fn=lambda idx: f"sun_{idx}.jpg",
                    num_examples=num_images_per_class,
                )
            )

        with open(data_dir / "ClassName.txt", "w") as file:
            file.writelines("\n".join(f"/{cls[0]}/{cls}" for cls in sampled_classes))

        num_samples = len(im_paths)

        return num_samples


class DTDTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.DTD
    FEATURE_TYPES = (PIL.Image.Image, int)

    SUPPORT_TV_IMAGE_DECODE = True

    ADDITIONAL_CONFIGS = combinations_grid(
        split=("train", "test", "val"),
        # There is no need to test the whole matrix here, since each fold is treated exactly the same
        partition=(1, 5, 10),
    )

    def inject_fake_data(self, tmpdir: str, config):
        data_folder = pathlib.Path(tmpdir) / "dtd" / "dtd"

        num_images_per_class = 3
        image_folder = data_folder / "images"
        image_files = []
        for cls in ("banded", "marbled", "zigzagged"):
            image_files.extend(
                datasets_utils.create_image_folder(
                    image_folder,
                    cls,
                    file_name_fn=lambda idx: f"{cls}_{idx:04d}.jpg",
                    num_examples=num_images_per_class,
                )
            )

        meta_folder = data_folder / "labels"
        meta_folder.mkdir()
        image_ids = [str(path.relative_to(path.parents[1])).replace(os.sep, "/") for path in image_files]
        image_ids_in_config = random.choices(image_ids, k=len(image_files) // 2)
        with open(meta_folder / f"{config['split']}{config['partition']}.txt", "w") as file:
            file.write("\n".join(image_ids_in_config) + "\n")

        return len(image_ids_in_config)


class FER2013TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FER2013
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))

    FEATURE_TYPES = (PIL.Image.Image, (int, type(None)))

    def inject_fake_data(self, tmpdir, config):
        base_folder = os.path.join(tmpdir, "fer2013")
        os.makedirs(base_folder)

        use_icml = config.pop("use_icml", False)
        use_fer = config.pop("use_fer", False)

        num_samples = 5

        if use_icml or use_fer:
            pixels_key, usage_key = (" pixels", " Usage") if use_icml else ("pixels", "Usage")
            fieldnames = ("emotion", usage_key, pixels_key) if use_icml else ("emotion", pixels_key, usage_key)
            filename = "icml_face_data.csv" if use_icml else "fer2013.csv"
            with open(os.path.join(base_folder, filename), "w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_NONNUMERIC,
                    quotechar='"',
                )
                writer.writeheader()
                for i in range(num_samples):
                    row = {
                        "emotion": str(int(torch.randint(0, 7, ()))),
                        usage_key: "Training" if i % 2 else "PublicTest",
                        pixels_key: " ".join(
                            str(pixel)
                            for pixel in datasets_utils.create_image_or_video_tensor((48, 48)).view(-1).tolist()
                        ),
                    }

                    writer.writerow(row)
        else:
            with open(os.path.join(base_folder, f"{config['split']}.csv"), "w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("emotion", "pixels") if config["split"] == "train" else ("pixels",),
                    quoting=csv.QUOTE_NONNUMERIC,
                    quotechar='"',
                )
                writer.writeheader()
                for _ in range(num_samples):
                    row = dict(
                        pixels=" ".join(
                            str(pixel)
                            for pixel in datasets_utils.create_image_or_video_tensor((48, 48)).view(-1).tolist()
                        )
                    )
                    if config["split"] == "train":
                        row["emotion"] = str(int(torch.randint(0, 7, ())))

                    writer.writerow(row)

        return num_samples

    def test_icml_file(self):
        config = {"split": "test"}
        with self.create_dataset(config=config) as (dataset, _):
            assert all(s[1] is None for s in dataset)

        for split in ("train", "test"):
            for d in ({"use_icml": True}, {"use_fer": True}):
                config = {"split": split, **d}
                with self.create_dataset(config=config) as (dataset, _):
                    assert all(s[1] is not None for s in dataset)


class GTSRBTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.GTSRB
    FEATURE_TYPES = (PIL.Image.Image, int)

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))

    def inject_fake_data(self, tmpdir: str, config):
        root_folder = os.path.join(tmpdir, "gtsrb")
        os.makedirs(root_folder, exist_ok=True)

        # Train data
        train_folder = os.path.join(root_folder, "GTSRB", "Training")
        os.makedirs(train_folder, exist_ok=True)

        num_examples = 3 if config["split"] == "train" else 4
        classes = ("00000", "00042", "00012")
        for class_idx in classes:
            datasets_utils.create_image_folder(
                train_folder,
                name=class_idx,
                file_name_fn=lambda image_idx: f"{class_idx}_{image_idx:05d}.ppm",
                num_examples=num_examples,
            )

        total_number_of_examples = num_examples * len(classes)
        # Test data
        test_folder = os.path.join(root_folder, "GTSRB", "Final_Test", "Images")
        os.makedirs(test_folder, exist_ok=True)

        with open(os.path.join(root_folder, "GT-final_test.csv"), "w") as csv_file:
            csv_file.write("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n")

            for _ in range(total_number_of_examples):
                image_file = datasets_utils.create_random_string(5, string.digits) + ".ppm"
                datasets_utils.create_image_file(test_folder, image_file)
                row = [
                    image_file,
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(1, 100, size=()).item(),
                    torch.randint(0, 43, size=()).item(),
                ]
                csv_file.write(";".join(map(str, row)) + "\n")

        return total_number_of_examples


class CLEVRClassificationTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CLEVRClassification
    FEATURE_TYPES = (PIL.Image.Image, (int, type(None)))

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))
    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        data_folder = pathlib.Path(tmpdir) / "clevr" / "CLEVR_v1.0"

        images_folder = data_folder / "images"
        image_files = datasets_utils.create_image_folder(
            images_folder, config["split"], lambda idx: f"CLEVR_{config['split']}_{idx:06d}.png", num_examples=5
        )

        scenes_folder = data_folder / "scenes"
        scenes_folder.mkdir()
        if config["split"] != "test":
            with open(scenes_folder / f"CLEVR_{config['split']}_scenes.json", "w") as file:
                json.dump(
                    dict(
                        info=dict(),
                        scenes=[
                            dict(image_filename=image_file.name, objects=[dict()] * int(torch.randint(10, ())))
                            for image_file in image_files
                        ],
                    ),
                    file,
                )

        return len(image_files)


class OxfordIIITPetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.OxfordIIITPet
    FEATURE_TYPES = (PIL.Image.Image, (int, PIL.Image.Image, tuple, type(None)))

    ADDITIONAL_CONFIGS = combinations_grid(
        split=("trainval", "test"),
        target_types=("category", "binary-category", "segmentation", ["category", "segmentation"], []),
    )

    def inject_fake_data(self, tmpdir, config):
        base_folder = os.path.join(tmpdir, "oxford-iiit-pet")

        classification_anns_meta = (
            dict(cls="Abyssinian", label=0, species="cat"),
            dict(cls="Keeshond", label=18, species="dog"),
            dict(cls="Yorkshire Terrier", label=37, species="dog"),
        )
        split_and_classification_anns = [
            self._meta_to_split_and_classification_ann(meta, idx)
            for meta, idx in itertools.product(classification_anns_meta, (1, 2, 10))
        ]
        image_ids, *_ = zip(*split_and_classification_anns)

        image_files = datasets_utils.create_image_folder(
            base_folder, "images", file_name_fn=lambda idx: f"{image_ids[idx]}.jpg", num_examples=len(image_ids)
        )

        anns_folder = os.path.join(base_folder, "annotations")
        os.makedirs(anns_folder)
        split_and_classification_anns_in_split = random.choices(split_and_classification_anns, k=len(image_ids) // 2)
        with open(os.path.join(anns_folder, f"{config['split']}.txt"), "w", newline="") as file:
            writer = csv.writer(file, delimiter=" ")
            for split_and_classification_ann in split_and_classification_anns_in_split:
                writer.writerow(split_and_classification_ann)

        segmentation_files = datasets_utils.create_image_folder(
            anns_folder, "trimaps", file_name_fn=lambda idx: f"{image_ids[idx]}.png", num_examples=len(image_ids)
        )

        # The dataset has some rogue files
        for path in image_files[:2]:
            path.with_suffix(".mat").touch()
        for path in segmentation_files:
            path.with_name(f".{path.name}").touch()

        return len(split_and_classification_anns_in_split)

    def _meta_to_split_and_classification_ann(self, meta, idx):
        image_id = "_".join(
            [
                *[(str.title if meta["species"] == "cat" else str.lower)(part) for part in meta["cls"].split()],
                str(idx),
            ]
        )
        class_id = str(meta["label"] + 1)
        species = "1" if meta["species"] == "cat" else "2"
        breed_id = "-1"
        return (image_id, class_id, species, breed_id)

    def test_transforms_v2_wrapper_spawn(self):
        expected_size = (123, 321)
        with self.create_dataset(transform=v2.Resize(size=expected_size)) as (dataset, _):
            datasets_utils.check_transforms_v2_wrapper_spawn(dataset, expected_size=expected_size)


class StanfordCarsTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.StanfordCars
    REQUIRED_PACKAGES = ("scipy",)
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir, config):
        import scipy.io as io
        from numpy.core.records import fromarrays

        num_examples = {"train": 5, "test": 7}[config["split"]]
        num_classes = 3
        base_folder = pathlib.Path(tmpdir) / "stanford_cars"

        devkit = base_folder / "devkit"
        devkit.mkdir(parents=True)

        if config["split"] == "train":
            images_folder_name = "cars_train"
            annotations_mat_path = devkit / "cars_train_annos.mat"
        else:
            images_folder_name = "cars_test"
            annotations_mat_path = base_folder / "cars_test_annos_withlabels.mat"

        datasets_utils.create_image_folder(
            root=base_folder,
            name=images_folder_name,
            file_name_fn=lambda image_index: f"{image_index:5d}.jpg",
            num_examples=num_examples,
        )

        classes = np.random.randint(1, num_classes + 1, num_examples, dtype=np.uint8)
        fnames = [f"{i:5d}.jpg" for i in range(num_examples)]
        rec_array = fromarrays(
            [classes, fnames],
            names=["class", "fname"],
        )
        io.savemat(annotations_mat_path, {"annotations": rec_array})

        random_class_names = ["random_name"] * num_classes
        io.savemat(devkit / "cars_meta.mat", {"class_names": random_class_names})

        return num_examples


class Country211TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Country211

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "valid", "test"))

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        split_folder = pathlib.Path(tmpdir) / "country211" / config["split"]
        split_folder.mkdir(parents=True, exist_ok=True)

        num_examples = {
            "train": 3,
            "valid": 4,
            "test": 5,
        }[config["split"]]

        classes = ("AD", "BS", "GR")
        for cls in classes:
            datasets_utils.create_image_folder(
                split_folder,
                name=cls,
                file_name_fn=lambda idx: f"{idx}.jpg",
                num_examples=num_examples,
            )

        return num_examples * len(classes)


class Flowers102TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Flowers102

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))
    REQUIRED_PACKAGES = ("scipy",)

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        base_folder = pathlib.Path(tmpdir) / "flowers-102"

        num_classes = 3
        num_images_per_split = dict(train=5, val=4, test=3)
        num_images_total = sum(num_images_per_split.values())
        datasets_utils.create_image_folder(
            base_folder,
            "jpg",
            file_name_fn=lambda idx: f"image_{idx + 1:05d}.jpg",
            num_examples=num_images_total,
        )

        label_dict = dict(
            labels=np.random.randint(1, num_classes + 1, size=(1, num_images_total), dtype=np.uint8),
        )
        datasets_utils.lazy_importer.scipy.io.savemat(str(base_folder / "imagelabels.mat"), label_dict)

        setid_mat = np.arange(1, num_images_total + 1, dtype=np.uint16)
        np.random.shuffle(setid_mat)
        setid_dict = dict(
            trnid=setid_mat[: num_images_per_split["train"]].reshape(1, -1),
            valid=setid_mat[num_images_per_split["train"] : -num_images_per_split["test"]].reshape(1, -1),
            tstid=setid_mat[-num_images_per_split["test"] :].reshape(1, -1),
        )
        datasets_utils.lazy_importer.scipy.io.savemat(str(base_folder / "setid.mat"), setid_dict)

        return num_images_per_split[config["split"]]


class PCAMTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.PCAM

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))
    REQUIRED_PACKAGES = ("h5py",)

    def inject_fake_data(self, tmpdir: str, config):
        base_folder = pathlib.Path(tmpdir) / "pcam"
        base_folder.mkdir()

        num_images = {"train": 2, "test": 3, "val": 4}[config["split"]]

        images_file = datasets.PCAM._FILES[config["split"]]["images"][0]
        with datasets_utils.lazy_importer.h5py.File(str(base_folder / images_file), "w") as f:
            f["x"] = np.random.randint(0, 256, size=(num_images, 10, 10, 3), dtype=np.uint8)

        targets_file = datasets.PCAM._FILES[config["split"]]["targets"][0]
        with datasets_utils.lazy_importer.h5py.File(str(base_folder / targets_file), "w") as f:
            f["y"] = np.random.randint(0, 2, size=(num_images, 1, 1, 1), dtype=np.uint8)

        return num_images


class RenderedSST2TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.RenderedSST2
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))
    SPLIT_TO_FOLDER = {"train": "train", "val": "valid", "test": "test"}

    SUPPORT_TV_IMAGE_DECODE = True

    def inject_fake_data(self, tmpdir: str, config):
        root_folder = pathlib.Path(tmpdir) / "rendered-sst2"
        image_folder = root_folder / self.SPLIT_TO_FOLDER[config["split"]]

        num_images_per_class = {"train": 5, "test": 6, "val": 7}
        sampled_classes = ["positive", "negative"]
        for cls in sampled_classes:
            datasets_utils.create_image_folder(
                image_folder,
                cls,
                file_name_fn=lambda idx: f"{idx}.png",
                num_examples=num_images_per_class[config["split"]],
            )

        return len(sampled_classes) * num_images_per_class[config["split"]]


class Kitti2012StereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Kitti2012Stereo
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    def inject_fake_data(self, tmpdir, config):
        kitti_dir = pathlib.Path(tmpdir) / "Kitti2012"
        os.makedirs(kitti_dir, exist_ok=True)

        split_dir = kitti_dir / (config["split"] + "ing")
        os.makedirs(split_dir, exist_ok=True)

        num_examples = {"train": 4, "test": 3}.get(config["split"], 0)

        datasets_utils.create_image_folder(
            root=split_dir,
            name="colored_0",
            file_name_fn=lambda i: f"{i:06d}_10.png",
            num_examples=num_examples,
            size=(3, 100, 200),
        )
        datasets_utils.create_image_folder(
            root=split_dir,
            name="colored_1",
            file_name_fn=lambda i: f"{i:06d}_10.png",
            num_examples=num_examples,
            size=(3, 100, 200),
        )

        if config["split"] == "train":
            datasets_utils.create_image_folder(
                root=split_dir,
                name="disp_noc",
                file_name_fn=lambda i: f"{i:06d}.png",
                num_examples=num_examples,
                # Kitti2012 uses a single channel image for disparities
                size=(1, 100, 200),
            )

        return num_examples

    def test_train_splits(self):
        for split in ["train"]:
            with self.create_dataset(split=split) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    assert mask is None
                    datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_test_split(self):
        for split in ["test"]:
            with self.create_dataset(split=split) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    assert mask is None
                    assert disparity is None
                    datasets_utils.shape_test_for_stereo(left, right)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass


class Kitti2015StereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Kitti2015Stereo
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    def inject_fake_data(self, tmpdir, config):
        kitti_dir = pathlib.Path(tmpdir) / "Kitti2015"
        os.makedirs(kitti_dir, exist_ok=True)

        split_dir = kitti_dir / (config["split"] + "ing")
        os.makedirs(split_dir, exist_ok=True)

        num_examples = {"train": 4, "test": 6}.get(config["split"], 0)

        datasets_utils.create_image_folder(
            root=split_dir,
            name="image_2",
            file_name_fn=lambda i: f"{i:06d}_10.png",
            num_examples=num_examples,
            size=(3, 100, 200),
        )
        datasets_utils.create_image_folder(
            root=split_dir,
            name="image_3",
            file_name_fn=lambda i: f"{i:06d}_10.png",
            num_examples=num_examples,
            size=(3, 100, 200),
        )

        if config["split"] == "train":
            datasets_utils.create_image_folder(
                root=split_dir,
                name="disp_occ_0",
                file_name_fn=lambda i: f"{i:06d}.png",
                num_examples=num_examples,
                # Kitti2015 uses a single channel image for disparities
                size=(1, 100, 200),
            )

            datasets_utils.create_image_folder(
                root=split_dir,
                name="disp_occ_1",
                file_name_fn=lambda i: f"{i:06d}.png",
                num_examples=num_examples,
                # Kitti2015 uses a single channel image for disparities
                size=(1, 100, 200),
            )

        return num_examples

    def test_train_splits(self):
        for split in ["train"]:
            with self.create_dataset(split=split) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    assert mask is None
                    datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_test_split(self):
        for split in ["test"]:
            with self.create_dataset(split=split) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    assert mask is None
                    assert disparity is None
                    datasets_utils.shape_test_for_stereo(left, right)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass


class CarlaStereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CarlaStereo
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, None))

    @staticmethod
    def _create_scene_folders(num_examples: int, root_dir: Union[str, pathlib.Path]):
        # make the root_dir if it does not exits
        os.makedirs(root_dir, exist_ok=True)

        for i in range(num_examples):
            scene_dir = pathlib.Path(root_dir) / f"scene_{i}"
            os.makedirs(scene_dir, exist_ok=True)
            # populate with left right images
            datasets_utils.create_image_file(root=scene_dir, name="im0.png", size=(100, 100))
            datasets_utils.create_image_file(root=scene_dir, name="im1.png", size=(100, 100))
            datasets_utils.make_fake_pfm_file(100, 100, file_name=str(scene_dir / "disp0GT.pfm"))
            datasets_utils.make_fake_pfm_file(100, 100, file_name=str(scene_dir / "disp1GT.pfm"))

    def inject_fake_data(self, tmpdir, config):
        carla_dir = pathlib.Path(tmpdir) / "carla-highres"
        os.makedirs(carla_dir, exist_ok=True)

        split_dir = pathlib.Path(carla_dir) / "trainingF"
        os.makedirs(split_dir, exist_ok=True)

        num_examples = 6
        self._create_scene_folders(num_examples=num_examples, root_dir=split_dir)

        return num_examples

    def test_train_splits(self):
        with self.create_dataset() as (dataset, _):
            for left, right, disparity in dataset:
                datasets_utils.shape_test_for_stereo(left, right, disparity)


class CREStereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CREStereo
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, np.ndarray, type(None))

    def inject_fake_data(self, tmpdir, config):
        crestereo_dir = pathlib.Path(tmpdir) / "CREStereo"
        os.makedirs(crestereo_dir, exist_ok=True)

        examples = {"tree": 2, "shapenet": 3, "reflective": 6, "hole": 5}

        for category_name in ["shapenet", "reflective", "tree", "hole"]:
            split_dir = crestereo_dir / category_name
            os.makedirs(split_dir, exist_ok=True)
            num_examples = examples[category_name]

            for idx in range(num_examples):
                datasets_utils.create_image_file(root=split_dir, name=f"{idx}_left.jpg", size=(100, 100))
                datasets_utils.create_image_file(root=split_dir, name=f"{idx}_right.jpg", size=(100, 100))
                # these are going to end up being gray scale images
                datasets_utils.create_image_file(root=split_dir, name=f"{idx}_left.disp.png", size=(1, 100, 100))
                datasets_utils.create_image_file(root=split_dir, name=f"{idx}_right.disp.png", size=(1, 100, 100))

        return sum(examples.values())

    def test_splits(self):
        with self.create_dataset() as (dataset, _):
            for left, right, disparity, mask in dataset:
                assert mask is None
                datasets_utils.shape_test_for_stereo(left, right, disparity)


class FallingThingsStereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FallingThingsStereo
    ADDITIONAL_CONFIGS = combinations_grid(variant=("single", "mixed", "both"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))

    @staticmethod
    def _make_dummy_depth_map(root: str, name: str, size: Tuple[int, int]):
        file = pathlib.Path(root) / name
        image = np.ones((size[0], size[1]), dtype=np.uint8)
        PIL.Image.fromarray(image).save(file)

    @staticmethod
    def _make_scene_folder(root: str, scene_name: str, size: Tuple[int, int]) -> None:
        root = pathlib.Path(root) / scene_name
        os.makedirs(root, exist_ok=True)
        # jpg images
        datasets_utils.create_image_file(root, "image1.left.jpg", size=(3, size[1], size[0]))
        datasets_utils.create_image_file(root, "image1.right.jpg", size=(3, size[1], size[0]))
        # single channel depth maps
        FallingThingsStereoTestCase._make_dummy_depth_map(root, "image1.left.depth.png", size=(size[0], size[1]))
        FallingThingsStereoTestCase._make_dummy_depth_map(root, "image1.right.depth.png", size=(size[0], size[1]))
        # camera settings json. Minimal example for _read_disparity function testing
        settings_json = {"camera_settings": [{"intrinsic_settings": {"fx": 1}}]}
        with open(root / "_camera_settings.json", "w") as f:
            json.dump(settings_json, f)

    def inject_fake_data(self, tmpdir, config):
        fallingthings_dir = pathlib.Path(tmpdir) / "FallingThings"
        os.makedirs(fallingthings_dir, exist_ok=True)

        num_examples = {"single": 2, "mixed": 3, "both": 4}.get(config["variant"], 0)

        variants = {
            "single": ["single"],
            "mixed": ["mixed"],
            "both": ["single", "mixed"],
        }.get(config["variant"], [])

        variant_dir_prefixes = {
            "single": 1,
            "mixed": 0,
        }

        for variant_name in variants:
            variant_dir = pathlib.Path(fallingthings_dir) / variant_name
            os.makedirs(variant_dir, exist_ok=True)

            for i in range(variant_dir_prefixes[variant_name]):
                variant_dir = variant_dir / f"{i:02d}"
                os.makedirs(variant_dir, exist_ok=True)

            for i in range(num_examples):
                self._make_scene_folder(
                    root=variant_dir,
                    scene_name=f"scene_{i:06d}",
                    size=(100, 200),
                )

        if config["variant"] == "both":
            num_examples *= 2
        return num_examples

    def test_splits(self):
        for variant_name in ["single", "mixed"]:
            with self.create_dataset(variant=variant_name) as (dataset, _):
                for left, right, disparity in dataset:
                    datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument variant"):
            with self.create_dataset(variant="bad"):
                pass


class SceneFlowStereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SceneFlowStereo
    ADDITIONAL_CONFIGS = combinations_grid(
        variant=("FlyingThings3D", "Driving", "Monkaa"), pass_name=("clean", "final", "both")
    )
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))

    @staticmethod
    def _create_pfm_folder(
        root: str, name: str, file_name_fn: Callable[..., str], num_examples: int, size: Tuple[int, int]
    ) -> None:
        root = pathlib.Path(root) / name
        os.makedirs(root, exist_ok=True)

        for i in range(num_examples):
            datasets_utils.make_fake_pfm_file(size[0], size[1], root / file_name_fn(i))

    def inject_fake_data(self, tmpdir, config):
        scene_flow_dir = pathlib.Path(tmpdir) / "SceneFlow"
        os.makedirs(scene_flow_dir, exist_ok=True)

        variant_dir = scene_flow_dir / config["variant"]
        variant_dir_prefixes = {
            "Monkaa": 0,
            "Driving": 2,
            "FlyingThings3D": 2,
        }
        os.makedirs(variant_dir, exist_ok=True)

        num_examples = {"FlyingThings3D": 4, "Driving": 6, "Monkaa": 5}.get(config["variant"], 0)

        passes = {
            "clean": ["frames_cleanpass"],
            "final": ["frames_finalpass"],
            "both": ["frames_cleanpass", "frames_finalpass"],
        }.get(config["pass_name"], [])

        for pass_dir_name in passes:
            # create pass directories
            pass_dir = variant_dir / pass_dir_name
            disp_dir = variant_dir / "disparity"
            os.makedirs(pass_dir, exist_ok=True)
            os.makedirs(disp_dir, exist_ok=True)

            for i in range(variant_dir_prefixes.get(config["variant"], 0)):
                pass_dir = pass_dir / str(i)
                disp_dir = disp_dir / str(i)
                os.makedirs(pass_dir, exist_ok=True)
                os.makedirs(disp_dir, exist_ok=True)

            for direction in ["left", "right"]:
                for scene_idx in range(num_examples):
                    os.makedirs(pass_dir / f"scene_{scene_idx:06d}", exist_ok=True)
                    datasets_utils.create_image_folder(
                        root=pass_dir / f"scene_{scene_idx:06d}",
                        name=direction,
                        file_name_fn=lambda i: f"{i:06d}.png",
                        num_examples=1,
                        size=(3, 200, 100),
                    )

                    os.makedirs(disp_dir / f"scene_{scene_idx:06d}", exist_ok=True)
                    self._create_pfm_folder(
                        root=disp_dir / f"scene_{scene_idx:06d}",
                        name=direction,
                        file_name_fn=lambda i: f"{i:06d}.pfm",
                        num_examples=1,
                        size=(100, 200),
                    )

        if config["pass_name"] == "both":
            num_examples *= 2
        return num_examples

    def test_splits(self):
        for variant_name, pass_name in itertools.product(["FlyingThings3D", "Driving", "Monkaa"], ["clean", "final"]):
            with self.create_dataset(variant=variant_name, pass_name=pass_name) as (dataset, _):
                for left, right, disparity in dataset:
                    datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument variant"):
            with self.create_dataset(variant="bad"):
                pass


class InStereo2k(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.InStereo2k
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)))
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))

    @staticmethod
    def _make_scene_folder(root: str, name: str, size: Tuple[int, int]):
        root = pathlib.Path(root) / name
        os.makedirs(root, exist_ok=True)

        datasets_utils.create_image_file(root=root, name="left.png", size=(3, size[0], size[1]))
        datasets_utils.create_image_file(root=root, name="right.png", size=(3, size[0], size[1]))
        datasets_utils.create_image_file(root=root, name="left_disp.png", size=(1, size[0], size[1]))
        datasets_utils.create_image_file(root=root, name="right_disp.png", size=(1, size[0], size[1]))

    def inject_fake_data(self, tmpdir, config):
        in_stereo_dir = pathlib.Path(tmpdir) / "InStereo2k"
        os.makedirs(in_stereo_dir, exist_ok=True)

        split_dir = pathlib.Path(in_stereo_dir) / config["split"]
        os.makedirs(split_dir, exist_ok=True)

        num_examples = {"train": 4, "test": 5}.get(config["split"], 0)

        for i in range(num_examples):
            self._make_scene_folder(split_dir, f"scene_{i:06d}", (100, 200))

        return num_examples

    def test_splits(self):
        for split_name in ["train", "test"]:
            with self.create_dataset(split=split_name) as (dataset, _):
                for left, right, disparity in dataset:
                    datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_bad_input(self):
        with pytest.raises(
            ValueError, match="Unknown value 'bad' for argument split. Valid values are {'train', 'test'}."
        ):
            with self.create_dataset(split="bad"):
                pass


class SintelStereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SintelStereo
    ADDITIONAL_CONFIGS = combinations_grid(pass_name=("final", "clean", "both"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    def inject_fake_data(self, tmpdir, config):
        sintel_dir = pathlib.Path(tmpdir) / "Sintel"
        os.makedirs(sintel_dir, exist_ok=True)

        split_dir = pathlib.Path(sintel_dir) / "training"
        os.makedirs(split_dir, exist_ok=True)

        # a single setting, since there are no splits
        num_examples = {"final": 2, "clean": 3}
        pass_names = {
            "final": ["final"],
            "clean": ["clean"],
            "both": ["final", "clean"],
        }.get(config["pass_name"], [])

        for p in pass_names:
            for view in [f"{p}_left", f"{p}_right"]:
                root = split_dir / view
                os.makedirs(root, exist_ok=True)

                datasets_utils.create_image_folder(
                    root=root,
                    name="scene1",
                    file_name_fn=lambda i: f"{i:06d}.png",
                    num_examples=num_examples[p],
                    size=(3, 100, 200),
                )

        datasets_utils.create_image_folder(
            root=split_dir / "occlusions",
            name="scene1",
            file_name_fn=lambda i: f"{i:06d}.png",
            num_examples=max(num_examples.values()),
            size=(1, 100, 200),
        )

        datasets_utils.create_image_folder(
            root=split_dir / "outofframe",
            name="scene1",
            file_name_fn=lambda i: f"{i:06d}.png",
            num_examples=max(num_examples.values()),
            size=(1, 100, 200),
        )

        datasets_utils.create_image_folder(
            root=split_dir / "disparities",
            name="scene1",
            file_name_fn=lambda i: f"{i:06d}.png",
            num_examples=max(num_examples.values()),
            size=(3, 100, 200),
        )

        if config["pass_name"] == "both":
            num_examples = sum(num_examples.values())
        else:
            num_examples = num_examples.get(config["pass_name"], 0)

        return num_examples

    def test_splits(self):
        for pass_name in ["final", "clean", "both"]:
            with self.create_dataset(pass_name=pass_name) as (dataset, _):
                for left, right, disparity, valid_mask in dataset:
                    datasets_utils.shape_test_for_stereo(left, right, disparity, valid_mask)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument pass_name"):
            with self.create_dataset(pass_name="bad"):
                pass


class ETH3DStereoestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ETH3DStereo
    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "test"))
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    @staticmethod
    def _create_scene_folder(num_examples: int, root_dir: str):
        # make the root_dir if it does not exits
        root_dir = pathlib.Path(root_dir)
        os.makedirs(root_dir, exist_ok=True)

        for i in range(num_examples):
            scene_dir = root_dir / f"scene_{i}"
            os.makedirs(scene_dir, exist_ok=True)
            # populate with left right images
            datasets_utils.create_image_file(root=scene_dir, name="im0.png", size=(100, 100))
            datasets_utils.create_image_file(root=scene_dir, name="im1.png", size=(100, 100))

    @staticmethod
    def _create_annotation_folder(num_examples: int, root_dir: str):
        # make the root_dir if it does not exits
        root_dir = pathlib.Path(root_dir)
        os.makedirs(root_dir, exist_ok=True)

        # create scene directories
        for i in range(num_examples):
            scene_dir = root_dir / f"scene_{i}"
            os.makedirs(scene_dir, exist_ok=True)
            # populate with a random png file for occlusion mask, and a pfm file for disparity
            datasets_utils.create_image_file(root=scene_dir, name="mask0nocc.png", size=(1, 100, 100))

            pfm_path = scene_dir / "disp0GT.pfm"
            datasets_utils.make_fake_pfm_file(h=100, w=100, file_name=pfm_path)

    def inject_fake_data(self, tmpdir, config):
        eth3d_dir = pathlib.Path(tmpdir) / "ETH3D"

        num_examples = 2 if config["split"] == "train" else 3

        split_name = "two_view_training" if config["split"] == "train" else "two_view_test"
        split_dir = eth3d_dir / split_name
        self._create_scene_folder(num_examples, split_dir)

        if config["split"] == "train":
            annot_dir = eth3d_dir / "two_view_training_gt"
            self._create_annotation_folder(num_examples, annot_dir)

        return num_examples

    def test_training_splits(self):
        with self.create_dataset(split="train") as (dataset, _):
            for left, right, disparity, valid_mask in dataset:
                datasets_utils.shape_test_for_stereo(left, right, disparity, valid_mask)

    def test_testing_splits(self):
        with self.create_dataset(split="test") as (dataset, _):
            assert all(d == (None, None) for d in dataset._disparities)
            for left, right, disparity, valid_mask in dataset:
                assert valid_mask is None
                datasets_utils.shape_test_for_stereo(left, right, disparity)

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass


class Middlebury2014StereoTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Middlebury2014Stereo
    ADDITIONAL_CONFIGS = combinations_grid(
        split=("train", "additional"),
        calibration=("perfect", "imperfect", "both"),
        use_ambient_views=(True, False),
    )
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image, (np.ndarray, type(None)), (np.ndarray, type(None)))

    @staticmethod
    def _make_scene_folder(root_dir: str, scene_name: str, split: str) -> None:
        calibrations = [None] if split == "test" else ["-perfect", "-imperfect"]
        root_dir = pathlib.Path(root_dir)

        for c in calibrations:
            scene_dir = root_dir / f"{scene_name}{c}"
            os.makedirs(scene_dir, exist_ok=True)
            # make normal images first
            datasets_utils.create_image_file(root=scene_dir, name="im0.png", size=(3, 100, 100))
            datasets_utils.create_image_file(root=scene_dir, name="im1.png", size=(3, 100, 100))
            datasets_utils.create_image_file(root=scene_dir, name="im1E.png", size=(3, 100, 100))
            datasets_utils.create_image_file(root=scene_dir, name="im1L.png", size=(3, 100, 100))
            # these are going to end up being gray scale images
            datasets_utils.make_fake_pfm_file(h=100, w=100, file_name=scene_dir / "disp0.pfm")
            datasets_utils.make_fake_pfm_file(h=100, w=100, file_name=scene_dir / "disp1.pfm")

    def inject_fake_data(self, tmpdir, config):
        split_scene_map = {
            "train": ["Adirondack", "Jadeplant", "Motorcycle", "Piano"],
            "additional": ["Backpack", "Bicycle1", "Cable", "Classroom1"],
            "test": ["Plants", "Classroom2E", "Classroom2", "Australia"],
        }

        middlebury_dir = pathlib.Path(tmpdir, "Middlebury2014")
        os.makedirs(middlebury_dir, exist_ok=True)

        split_dir = middlebury_dir / config["split"]
        os.makedirs(split_dir, exist_ok=True)

        num_examples = {"train": 2, "additional": 3, "test": 4}.get(config["split"], 0)
        for idx in range(num_examples):
            scene_name = split_scene_map[config["split"]][idx]
            self._make_scene_folder(root_dir=split_dir, scene_name=scene_name, split=config["split"])

        if config["calibration"] == "both":
            num_examples *= 2
        return num_examples

    def test_train_splits(self):
        for split, calibration in itertools.product(["train", "additional"], ["perfect", "imperfect", "both"]):
            with self.create_dataset(split=split, calibration=calibration) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    datasets_utils.shape_test_for_stereo(left, right, disparity, mask)

    def test_test_split(self):
        for split in ["test"]:
            with self.create_dataset(split=split, calibration=None) as (dataset, _):
                for left, right, disparity, mask in dataset:
                    datasets_utils.shape_test_for_stereo(left, right)

    def test_augmented_view_usage(self):
        with self.create_dataset(split="train", use_ambient_views=True) as (dataset, _):
            for left, right, disparity, mask in dataset:
                datasets_utils.shape_test_for_stereo(left, right, disparity, mask)

    def test_value_err_train(self):
        # train set invalid
        split = "train"
        calibration = None
        with pytest.raises(
            ValueError,
            match=f"Split '{split}' has calibration settings, however None was provided as an argument."
            f"\nSetting calibration to 'perfect' for split '{split}'. Available calibration settings are: 'perfect', 'imperfect', 'both'.",
        ):
            with self.create_dataset(split=split, calibration=calibration):
                pass

    def test_value_err_test(self):
        # test set invalid
        split = "test"
        calibration = "perfect"
        with pytest.raises(
            ValueError, match="Split 'test' has only no calibration settings, please set `calibration=None`."
        ):
            with self.create_dataset(split=split, calibration=calibration):
                pass

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Unknown value 'bad' for argument split"):
            with self.create_dataset(split="bad"):
                pass


class ImagenetteTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Imagenette
    ADDITIONAL_CONFIGS = combinations_grid(split=["train", "val"], size=["full", "320px", "160px"])

    SUPPORT_TV_IMAGE_DECODE = True

    _WNIDS = [
        "n01440764",
        "n02102040",
        "n02979186",
        "n03000684",
        "n03028079",
        "n03394916",
        "n03417042",
        "n03425413",
        "n03445777",
        "n03888257",
    ]

    def inject_fake_data(self, tmpdir, config):
        archive_root = "imagenette2"
        if config["size"] != "full":
            archive_root += f"-{config['size'].replace('px', '')}"
        image_root = pathlib.Path(tmpdir) / archive_root / config["split"]

        num_images_per_class = 3
        for wnid in self._WNIDS:
            datasets_utils.create_image_folder(
                root=image_root,
                name=wnid,
                file_name_fn=lambda idx: f"{wnid}_{idx}.JPEG",
                num_examples=num_images_per_class,
            )

        return num_images_per_class * len(self._WNIDS)


class TestDatasetWrapper:
    def test_unknown_type(self):
        unknown_object = object()
        with pytest.raises(
            TypeError, match=re.escape("is meant for subclasses of `torchvision.datasets.VisionDataset`")
        ):
            datasets.wrap_dataset_for_transforms_v2(unknown_object)

    def test_unknown_dataset(self):
        class MyVisionDataset(datasets.VisionDataset):
            pass

        dataset = MyVisionDataset("root")

        with pytest.raises(TypeError, match="No wrapper exist"):
            datasets.wrap_dataset_for_transforms_v2(dataset)

    def test_missing_wrapper(self):
        dataset = datasets.FakeData()

        with pytest.raises(TypeError, match="please open an issue"):
            datasets.wrap_dataset_for_transforms_v2(dataset)

    def test_subclass(self, mocker):
        from torchvision import tv_tensors

        sentinel = object()
        mocker.patch.dict(
            tv_tensors._dataset_wrapper.WRAPPER_FACTORIES,
            clear=False,
            values={datasets.FakeData: lambda dataset, target_keys: lambda idx, sample: sentinel},
        )

        class MyFakeData(datasets.FakeData):
            pass

        dataset = MyFakeData()
        wrapped_dataset = datasets.wrap_dataset_for_transforms_v2(dataset)

        assert wrapped_dataset[0] is sentinel


if __name__ == "__main__":
    unittest.main()
