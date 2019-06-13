import os
import sys
import contextlib
import tarfile
import numpy as np
import PIL
import torch
from common_utils import get_tmp_dir

PYTHON2 = sys.version_info[0] == 2
if PYTHON2:
    import cPickle as pickle
else:
    import pickle


@contextlib.contextmanager
def mnist_root(num_images, cls_name):
    def _encode(v):
        return torch.tensor(v, dtype=torch.int32).numpy().tobytes()[::-1]

    def _make_image_file(filename, num_images):
        img = torch.randint(0, 255, size=(28 * 28 * num_images,), dtype=torch.uint8)
        with open(filename, "wb") as f:
            f.write(_encode(2051))  # magic header
            f.write(_encode(num_images))
            f.write(_encode(28))
            f.write(_encode(28))
            f.write(img.numpy().tobytes())

    def _make_label_file(filename, num_images):
        labels = torch.randint(0, 10, size=(num_images,), dtype=torch.uint8)
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
def imagenet_root():
    import scipy.io as sio

    WNID = 'n01234567'
    CLS = 'fakedata'

    def _make_image(file):
        PIL.Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(file)

    def _make_tar(archive, content, arcname=None, compress=False):
        mode = 'w:gz' if compress else 'w'
        if arcname is None:
            arcname = os.path.basename(content)
        with tarfile.open(archive, mode) as fh:
            fh.add(content, arcname=arcname)

    def _make_train_archive(root):
        with get_tmp_dir() as tmp:
            wnid_dir = os.path.join(tmp, WNID)
            os.mkdir(wnid_dir)

            _make_image(os.path.join(wnid_dir, WNID + '_1.JPEG'))

            wnid_archive = wnid_dir + '.tar'
            _make_tar(wnid_archive, wnid_dir)

            train_archive = os.path.join(root, 'ILSVRC2012_img_train.tar')
            _make_tar(train_archive, wnid_archive)

    def _make_val_archive(root):
        with get_tmp_dir() as tmp:
            val_image = os.path.join(tmp, 'ILSVRC2012_val_00000001.JPEG')
            _make_image(val_image)

            val_archive = os.path.join(root, 'ILSVRC2012_img_val.tar')
            _make_tar(val_archive, val_image)

    def _make_devkit_archive(root):
        with get_tmp_dir() as tmp:
            data_dir = os.path.join(tmp, 'data')
            os.mkdir(data_dir)

            meta_file = os.path.join(data_dir, 'meta.mat')
            synsets = np.core.records.fromarrays([
                (0.0, 1.0),
                (WNID, ''),
                (CLS, ''),
                ('fakedata for the torchvision testsuite', ''),
                (0.0, 1.0),
            ], names=['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children'])
            sio.savemat(meta_file, {'synsets': synsets})

            groundtruth_file = os.path.join(data_dir,
                                            'ILSVRC2012_validation_ground_truth.txt')
            with open(groundtruth_file, 'w') as fh:
                fh.write('0\n')

            devkit_name = 'ILSVRC2012_devkit_t12'
            devkit_archive = os.path.join(root, devkit_name + '.tar.gz')
            _make_tar(devkit_archive, tmp, arcname=devkit_name, compress=True)

    with get_tmp_dir() as root:
        _make_train_archive(root)
        _make_val_archive(root)
        _make_devkit_archive(root)

        yield root
