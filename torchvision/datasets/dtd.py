import os
from .folder import ImageFolder
from torch.utils.data import Subset
from .utils import download_url, check_integrity


class FullDTD(ImageFolder):
    """Full `DTD <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``dtd`` exists.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    image_folder = os.path.join('dtd', 'images')
    label_folder = os.path.join('dtd', 'labels')
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
    filename = 'dtd-r1.0.1.tar.gz'
    tgz_md5 = 'fff73e5086ae6bdbea199a49dfb8a4c1'

    def __init__(self, root, download=False, **kwargs):
        root = self.root = os.path.expanduser(root)

        if download:
            self.download()

        super().__init__(os.path.join(self.root, self.image_folder), **kwargs)
        # super class sets this to the root of the image folder, which is inside
        # the data folder
        self.root = root

    def download(self):
        import tarfile

        if not check_integrity(os.path.join(self.root, self.filename),
                               self.tgz_md5):
            download_url(self.url, self.root, self.filename,
                         self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.root, self.filename),
                           "r:gz")
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(
                                                                           tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DTD(Subset):
    """`DTD <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_ Dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``dtd`` exists.
            split (string, optional): The image split to use, ``train``, ``test``
                or ``val``
            fold (int, optional): The image fold to use, ``[1 ... 10]``
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
            transform (callable, optional): A function/transform that  takes in an
                PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
            loader (callable, optional): A function to load an image given its path.
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            split (string): image split
            fold (int): image fold
        """
    def __init__(self, root, split='train', fold=1, **kwargs):
        assert split in ('train', 'val', 'test'), \
            "split should be train, val or test"
        self.split = split

        assert fold in range(1, 11), "fold should be integer in [1, 10]"
        self.fold = fold

        dataset = FullDTD(root, **kwargs)
        indices = self._make_indices(dataset)
        super().__init__(dataset, indices)

    def _make_indices(self, dataset):
        image_folder = os.path.join(dataset.root, dataset.image_folder)
        image_paths = [path for path, target in dataset.imgs]

        label_folder = os.path.join(dataset.root, dataset.label_folder)
        file_name = '{}{}.txt'.format(self.split, self.fold)
        file_path = os.path.join(label_folder, file_name)
        with open(file_path, 'r') as f:
            image_paths_subset = f.read()
        image_paths_subset = [os.path.join(image_folder, path)
                              for path in image_paths_subset.splitlines()]

        return [image_paths.index(image_path_subset) for
                image_path_subset in image_paths_subset]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Fold: {}\n'.format(self.fold)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.dataset.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(
                                                                           tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.dataset.target_transform.__repr__().replace(
                                       '\n', '\n' + ' ' * len(tmp)))
        return fmt_str



