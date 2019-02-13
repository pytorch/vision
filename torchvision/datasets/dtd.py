import os
from .folder import ImageFolder
from .utils import download_url, check_integrity


class DTD(ImageFolder):
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
        imgs (list): List of (image path, class_index) tuples of the complete
            dataset regardless of the image split and fold

    """
    image_folder = os.path.join('dtd', 'images')
    label_folder = os.path.join('dtd', 'labels')
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
    filename = 'dtd-r1.0.1.tar.gz'
    tgz_md5 = 'fff73e5086ae6bdbea199a49dfb8a4c1'

    def __init__(self, root, split='train', fold=1, download=False,
                 **kwargs):
        self.base_folder = os.path.expanduser(root)

        # the input validation is done outside the properties to avoid creation
        # of the index converter
        self._validate_split(split)
        self._validate_fold(fold)
        self._split = split
        self._fold = fold

        if download:
            self.download()

        super().__init__(os.path.join(self.base_folder, self.image_folder),
                         **kwargs)
        self._make_index_converter()

    def download(self):
        import tarfile

        if not check_integrity(os.path.join(self.base_folder, self.filename),
                               self.tgz_md5):
            download_url(self.url, self.base_folder, self.filename,
                         self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.base_folder, self.filename),
                           "r:gz")
        os.chdir(self.base_folder)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._validate_split(split)
        if split != self._split:
            self._split = split
            self._make_index_converter()

    def _validate_split(self, split):
        assert split in ('train', 'val', 'test'), \
            "split should be train, val or test"

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, fold):
        self._validate_fold(fold)
        if fold != self._fold:
            self._fold = fold
            self._make_index_converter()

    def _validate_fold(self, fold):
        assert fold in range(1, 11), "fold should be integer in [1, 10]"

    def _make_index_converter(self):
        image_paths = [path for path, target in self.imgs]

        file_name = '{}{}.txt'.format(self.split, self.fold)
        file_path = os.path.join(self.base_folder, self.label_folder, file_name)
        with open(file_path, 'r') as f:
            image_paths_partial = f.read()
        image_paths_partial = [os.path.join(self.base_folder,
                                            self.image_folder,
                                            path)
                               for path in image_paths_partial.splitlines()]
        self._index_converter = dict(
            [(idx, image_paths.index(image_paths_partial[idx]))
             for idx in range(len(image_paths_partial))])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target
                class.
        """
        return super().__getitem__(self._index_converter[index])

    def __len__(self):
        return len(self._index_converter)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Fold: {}\n'.format(self.fold)
        fmt_str += '    Root Location: {}\n'.format(self.base_folder)
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
