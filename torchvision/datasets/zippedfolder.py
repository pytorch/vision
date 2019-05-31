from .vision import VisionDataset
from .utils import _is_zip
from .folder import has_file_allowed_extension, is_image_file, IMG_EXTENSIONS

from PIL import Image

import os
import os.path
import sys


class ZippedImageFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way inside of a ZIP file: ::

        class_x/xxx.png
        class_x/xxy.png
        class_x/xxz.png

        class_y/123.png
        class_y/nsdf3.png
        class_y/asd932_.png

    Args:
        root (string): Path to ZIP file.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None):
        assert _is_zip(root)
        super(ZippedImageFolder, self).__init__(root)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = self.zip_loader
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(self.root, class_to_idx, self.extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def zip_loader(self, path):
        f = self.root_zip[path]
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            try:
                import accimage
                return accimage.Image(f)
            except IOError:
                pass   # fall through to PIL
        return Image.open(f).convert('RGB')

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = list({filename.split("/")[0] for filename in self.root_zip.keys() if "/" in filename})
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)

        for filename in sorted(self.root_zip.keys()):
            if "/" in filename:
                target = filename.split("/", 1)[0]
                item = (filename, class_to_idx[target])
                images.append(item)

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
