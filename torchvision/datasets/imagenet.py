from __future__ import print_function
from contextlib import contextmanager
import os
import shutil
import tempfile
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg

ARCHIVE_DICT = {
    'train': {
        'file': 'ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'val': {
        'file': 'ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'file': 'ILSVRC2012_devkit_t12.tar.gz',
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
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def download(self):
        def check_archive(archive_dict):
            archive = os.path.join(self.root, archive_dict["file"])
            md5 = archive_dict["md5"]
            if not check_integrity(archive, md5):
                self._raise_download_error()

            return archive

        if not check_integrity(self.meta_file):
            archive = check_archive(ARCHIVE_DICT['devkit'])
            parse_devkit(archive)

        if not os.path.isdir(self.split_folder):
            archive = check_archive(ARCHIVE_DICT[self.split])

            if self.split == 'train':
                parse_train_archive(archive)
            elif self.split == 'val':
                parse_val_archive(archive)
        else:
            msg = ("You set download=True, but a folder '{}' already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg.format(self.split))

    def _raise_download_error(self):
        # FIXME
        raise RuntimeError

    @property
    def meta_file(self):
        return os.path.join(self.root, "meta.bin")

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


@contextmanager
def tmpdir():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    except:
        shutil.rmtree(tmpdir)


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))


def parse_devkit(archive, meta_file=None):
    """

    Args:
        archive:
        meta_file:
    """
    # FIXME
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

    if meta_file is None:
        meta_file = os.path.join(os.path.basename(archive), "meta.bin")

    with tmpdir() as devkit_root:
        extract_archive(archive, devkit_root)

        idx_to_wnid, wnid_to_classes = parse_meta(devkit_root)
        val_idcs = parse_val_groundtruth(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

    torch.save((wnid_to_classes, val_wnids), meta_file)


def load_meta_file(root, filename="meta.bin"):
    file = os.path.join(root, filename)
    if check_integrity(file):
        return torch.load(file)
    else:
        # FIXME
        raise RuntimeError("Meta file not found.")


def parse_train_archive(archive, folder=None):
    # FIXME
    """

    Args:
        archive:
        folder:

    Returns:

    """
    if folder is None:
        folder = os.path.join(os.path.basename(archive), "train")

    extract_archive(archive, folder)

    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(archive, wnids=None, folder=None):
    # FIXME
    """

    Args:
        archive:
        wnids:
        folder:
    """
    root = os.path.basename(archive)
    if wnids is None:
        wnids = load_meta_file(root)[1]
    if folder is None:
        folder = os.path.join(root, "val")

    extract_archive(archive, folder)

    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))
