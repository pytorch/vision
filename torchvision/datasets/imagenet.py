import warnings
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

META_FILE_NAME = "meta.bin"


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
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

    def __init__(self, root, split='train', download=None, **kwargs):
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.extract_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def extract_archives(self):
        if not check_integrity(self.meta_file):
            archive_dict = ARCHIVE_DICT['devkit']
            archive = os.path.join(self.root, archive_dict["file"])
            self._verify_archive(archive, archive_dict["md5"])

            parse_devkit_archive(archive)

        if not os.path.isdir(self.split_folder):
            archive_dict = ARCHIVE_DICT[self.split]
            archive = os.path.join(self.root, archive_dict["file"])
            self._verify_archive(archive, archive_dict["md5"])

            if self.split == 'train':
                parse_train_archive(archive)
            elif self.split == 'val':
                parse_val_archive(archive)

    def _verify_archive(self, archive, md5):
        if not check_integrity(archive, md5):
            msg = ("The file {} is not present in the root directory or corrupted. "
                   "You need to download it externally and place it in {}.")
            raise RuntimeError(msg.format(os.path.basename(archive), self.root))

    @property
    def meta_file(self):
        return os.path.join(self.root, META_FILE_NAME)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def parse_devkit_archive(archive, meta_file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        archive (str): Path to the devkit archive
        meta_file (str, optional): Optional name for the meta information file
    """
    import scipy.io as sio

    def parse_meta(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth(devkit_root):
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    if meta_file is None:
        meta_file = os.path.join(os.path.dirname(archive), META_FILE_NAME)

    tmpdir = tempfile.mkdtemp()
    extract_archive(archive, tmpdir)

    devkit_root = os.path.join(tmpdir, "ILSVRC2012_devkit_t12")
    idx_to_wnid, wnid_to_classes = parse_meta(devkit_root)
    val_idcs = parse_val_groundtruth(devkit_root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

    torch.save((wnid_to_classes, val_wnids), meta_file)

    shutil.rmtree(tmpdir)


def load_meta_file(root, filename=META_FILE_NAME):
    file = os.path.join(root, filename)
    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("Meta file not found at {}. You can create it with the "
               "parse_devkit_archive() function")
        raise RuntimeError(msg.format(file))


def parse_train_archive(archive, folder=None):
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        archive (str): Path to the train images archive
        folder (str, optional): Optional name for train images folder
    """
    if folder is None:
        folder = os.path.join(os.path.dirname(archive), "train")

    extract_archive(archive, folder)

    for archive in [os.path.join(folder, file) for file in os.listdir(folder)]:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(archive, wnids=None, folder=None):
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        archive (str): Path to the validation images archive
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are tried to be loaded from the meta information binary
            file in the same directory as the archive.
        folder (str, optional): Optional name for validation images folder
    """
    root = os.path.dirname(archive)
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


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
