from __future__ import print_function
import os
import shutil
import tempfile
import torch
from .folder import ImageFolder
from .utils import check_integrity, download_and_extract_archive, extract_archive, \
    verify_str_arg

ARCHIVE_DICT = {
    'train': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'val': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        if download:
            self.download()
        wnid_to_classes = self._load_meta_file()[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def download(self):
        if not check_integrity(self.meta_file):
            tmp_dir = tempfile.mkdtemp()

            archive_dict = ARCHIVE_DICT['devkit']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=tmp_dir,
                                         md5=archive_dict['md5'])
            devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta = parse_devkit(os.path.join(tmp_dir, devkit_folder))
            self._save_meta_file(*meta)

            shutil.rmtree(tmp_dir)

        if not os.path.isdir(self.split_folder):
            archive_dict = ARCHIVE_DICT[self.split]
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=self.split_folder,
                                         md5=archive_dict['md5'])

            if self.split == 'train':
                prepare_train_folder(self.split_folder)
            elif self.split == 'val':
                val_wnids = self._load_meta_file()[1]
                prepare_val_folder(self.split_folder, val_wnids)
        else:
            msg = ("You set download=True, but a folder '{}' already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg.format(self.split))

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError("Meta file not found or corrupted.",
                               "You can use download=True to create it.")

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def parse_devkit(root):
    idx_to_wnid, wnid_to_classes = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return wnid_to_classes, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
    import scipy.io as sio

    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
    with open(os.path.join(devkit_root, path, filename), 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
