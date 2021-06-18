import os
import contextlib
import hashlib
import pickle
import re
import tarfile
import unittest.mock
from distutils import dir_util

import numpy as np
import PIL
import torch

from common_utils import get_tmp_dir


def mock_class_attribute(stack, target, new):
    mock = unittest.mock.patch(target, new_callable=unittest.mock.PropertyMock, return_value=new)
    stack.enter_context(mock)
    return mock


def compute_md5(file):
    with open(file, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def make_tar(root, name, *files, compression=None):
    ext = ".tar"
    mode = "w"
    if compression is not None:
        ext = f"{ext}.{compression}"
        mode = f"{mode}:{compression}"

    name = os.path.splitext(name)[0] + ext
    archive = os.path.join(root, name)

    with tarfile.open(archive, mode) as fh:
        for file in files:
            fh.add(os.path.join(root, file), arcname=file)

    return name, compute_md5(archive)


def clean_dir(root, *keep):
    pattern = re.compile(f"({f')|('.join(keep)})")
    for file_or_dir in os.listdir(root):
        if pattern.search(file_or_dir):
            continue

        file_or_dir = os.path.join(root, file_or_dir)
        if os.path.isfile(file_or_dir):
            os.remove(file_or_dir)
        else:
            dir_util.remove_tree(file_or_dir)


@contextlib.contextmanager
def mnist_root(num_images, cls_name):
    def _encode(v):
        return torch.tensor(v, dtype=torch.int32).numpy().tobytes()[::-1]

    def _make_image_file(filename, num_images):
        img = torch.randint(0, 256, size=(28 * 28 * num_images,), dtype=torch.uint8)
        with open(filename, "wb") as f:
            f.write(_encode(2051))  # magic header
            f.write(_encode(num_images))
            f.write(_encode(28))
            f.write(_encode(28))
            f.write(img.numpy().tobytes())

    def _make_label_file(filename, num_images):
        labels = torch.zeros((num_images,), dtype=torch.uint8)
        with open(filename, "wb") as f:
            f.write(_encode(2049))  # magic header
            f.write(_encode(num_images))
            f.write(labels.numpy().tobytes())

    with get_tmp_dir() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, cls_name, "raw")
        os.makedirs(raw_dir)
        _make_image_file(os.path.join(raw_dir, "train-images-idx3-ubyte"), num_images)
        _make_label_file(os.path.join(raw_dir, "train-labels-idx1-ubyte"), num_images)
        _make_image_file(os.path.join(raw_dir, "t10k-images-idx3-ubyte"), num_images)
        _make_label_file(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"), num_images)
        yield tmp_dir


@contextlib.contextmanager
def cifar_root(version):
    def _get_version_params(version):
        if version == 'CIFAR10':
            return {
                'base_folder': 'cifar-10-batches-py',
                'train_files': ['data_batch_{}'.format(batch) for batch in range(1, 6)],
                'test_file': 'test_batch',
                'target_key': 'labels',
                'meta_file': 'batches.meta',
                'classes_key': 'label_names',
            }
        elif version == 'CIFAR100':
            return {
                'base_folder': 'cifar-100-python',
                'train_files': ['train'],
                'test_file': 'test',
                'target_key': 'fine_labels',
                'meta_file': 'meta',
                'classes_key': 'fine_label_names',
            }
        else:
            raise ValueError

    def _make_pickled_file(obj, file):
        with open(file, 'wb') as fh:
            pickle.dump(obj, fh, 2)

    def _make_data_file(file, target_key):
        obj = {
            'data': np.zeros((1, 32 * 32 * 3), dtype=np.uint8),
            target_key: [0]
        }
        _make_pickled_file(obj, file)

    def _make_meta_file(file, classes_key):
        obj = {
            classes_key: ['fakedata'],
        }
        _make_pickled_file(obj, file)

    params = _get_version_params(version)
    with get_tmp_dir() as root:
        base_folder = os.path.join(root, params['base_folder'])
        os.mkdir(base_folder)

        for file in list(params['train_files']) + [params['test_file']]:
            _make_data_file(os.path.join(base_folder, file), params['target_key'])

        _make_meta_file(os.path.join(base_folder, params['meta_file']),
                        params['classes_key'])

        yield root


@contextlib.contextmanager
def widerface_root():
    """
    Generates a dataset with the following folder structure and returns the path root:
    <root>
        └── widerface
            ├── wider_face_split
            ├── WIDER_train
            ├── WIDER_val
            └── WIDER_test

    The dataset consist of
      1 image for each dataset split (train, val, test) and annotation files
      for each split
    """

    def _make_image(file):
        PIL.Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(file)

    def _make_train_archive(root):
        extracted_dir = os.path.join(root, 'WIDER_train', 'images', '0--Parade')
        os.makedirs(extracted_dir)
        _make_image(os.path.join(extracted_dir, '0_Parade_marchingband_1_1.jpg'))

    def _make_val_archive(root):
        extracted_dir = os.path.join(root, 'WIDER_val', 'images', '0--Parade')
        os.makedirs(extracted_dir)
        _make_image(os.path.join(extracted_dir, '0_Parade_marchingband_1_2.jpg'))

    def _make_test_archive(root):
        extracted_dir = os.path.join(root, 'WIDER_test', 'images', '0--Parade')
        os.makedirs(extracted_dir)
        _make_image(os.path.join(extracted_dir, '0_Parade_marchingband_1_3.jpg'))

    def _make_annotations_archive(root):
        train_bbox_contents = '0--Parade/0_Parade_marchingband_1_1.jpg\n1\n449 330 122 149 0 0 0 0 0 0\n'
        val_bbox_contents = '0--Parade/0_Parade_marchingband_1_2.jpg\n1\n501 160 285 443 0 0 0 0 0 0\n'
        test_filelist_contents = '0--Parade/0_Parade_marchingband_1_3.jpg\n'
        extracted_dir = os.path.join(root, 'wider_face_split')
        os.mkdir(extracted_dir)

        # bbox training file
        bbox_file = os.path.join(extracted_dir, "wider_face_train_bbx_gt.txt")
        with open(bbox_file, "w") as txt_file:
            txt_file.write(train_bbox_contents)

        # bbox validation file
        bbox_file = os.path.join(extracted_dir, "wider_face_val_bbx_gt.txt")
        with open(bbox_file, "w") as txt_file:
            txt_file.write(val_bbox_contents)

        # test filelist file
        filelist_file = os.path.join(extracted_dir, "wider_face_test_filelist.txt")
        with open(filelist_file, "w") as txt_file:
            txt_file.write(test_filelist_contents)

    with get_tmp_dir() as root:
        root_base = os.path.join(root, "widerface")
        os.mkdir(root_base)
        _make_train_archive(root_base)
        _make_val_archive(root_base)
        _make_test_archive(root_base)
        _make_annotations_archive(root_base)

        yield root
