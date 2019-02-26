import os
import sys
if sys.version_info[0] == 2:
    # FIXME: I don't know if this is good pratice / robust
    FileExistsError = OSError

import shutil
import scipy.io as sio
import torch
from .folder import ImageFolder
from .utils import check_integrity, download_url

ARCHIVE_DICT = {
    ('2012', 'train'): {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    ('2012', 'val'): {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    ('2012', 'devkit'): {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}

META_DICT = {
    '2012': '5c2648af14b2ff44540504b860a81a79',
}

META_FILE = 'meta.bin'


class ImageNetClassification(ImageFolder):
    def __init__(self, root, split='val', year='2012', download=False, **kwargs):

        root = self.root = os.path.expanduser(root)
        self.split = self._verify_split(split)
        self.year = self._verify_year(year)

        if download:
            self.download()

        self.wnids, self.wnid_to_idx, classes, class_to_idx = self._load_meta()
        super(ImageNetClassification, self).__init__(self.split_folder, **kwargs)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx

    def download(self):
        self._prepare_tree()

        meta_file = os.path.join(self.year_folder, META_FILE)
        if not check_integrity(meta_file, META_DICT[self.year]):
            tmpdir = os.path.join(self.root, 'tmp')

            archive_dict = ARCHIVE_DICT[(self.year, 'devkit')]
            download_and_extract_tar(archive_dict['url'], self.root,
                                     extract_root=tmpdir,
                                     md5=archive_dict['md5'], gzip=True)
            devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta, val_wnids = parse_devkit(os.path.join(tmpdir, devkit_folder))
            torch.save((meta, val_wnids), meta_file)

            shutil.rmtree(tmpdir)

        archive_dict = ARCHIVE_DICT[(self.year, self.split)]
        download_and_extract_tar(archive_dict['url'], self.root,
                                 extract_root=self.split_folder,
                                 md5=archive_dict['md5'], gzip=False)

        if self.split == 'val':
            val_wnids = torch.load(meta_file)[1]
            prepare_val_folder(self.split_folder, val_wnids)

    def _load_meta(self):
        # TODO: verify meta file
        return torch.load(os.path.join(self.year_folder, META_FILE))[0]

    def _prepare_tree(self):
        try:
            os.makedirs(self.split_folder)
        except FileExistsError:
            shutil.rmtree(self.split_folder)

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val'

    def _verify_year(self, year):
        if year not in self.valid_years:
            msg = "Unknown year {} .".format(year)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_years))
            raise ValueError(msg)
        return year

    @property
    def valid_years(self):
        return '2012',

    @property
    def base_folder(self):
        return os.path.join(self.root, 'ILSVRC')

    @property
    def year_folder(self):
        return os.path.join(self.base_folder, self.year)

    @property
    def split_folder(self):
        return os.path.join(self.year_folder, self.split)

    @property
    def archive_dict(self):
        return ARCHIVE_DICT[(self.year, self.split)]


class ImageNetDetection():
    # TODO: implement
    pass


def download_and_extract_tar(url, download_root, extract_root=None, filename=None,
                             md5=None, gzip=False):
    import tarfile

    download_root = os.path.expanduser(download_root)
    if not extract_root:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    archive = os.path.join(download_root, filename)
    if not check_integrity(archive, md5):
        download_url(url, download_root, filename=filename, md5=md5)

    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(archive, mode) as tgzfh:
        tgzfh.extractall(path=extract_root)


def parse_devkit(root):
    # FIXME: generalize this for all years
    meta = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)

    wnid_to_idx = meta[1]
    idx_to_wnid = {val: key for key, val in wnid_to_idx.items()}
    val_wnids = [idx_to_wnid[val_idx] for val_idx in val_idcs]

    return meta, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
    # FIXME: generalize this for all years
    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(cls.split(', ')) for cls in classes]
    wnid_to_idx = {wnid: idx for wnid, idx in zip(wnids, idcs)}
    class_to_idx = {cls: idx for cls, idx in zip(classes, idcs)}
    return wnids, wnid_to_idx, classes, class_to_idx


def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
    # FIXME: generalize this for all years
    with open(os.path.join(devkit_root, path, filename), 'r') as fh:
        val_idcs = fh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


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
