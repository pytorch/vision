import json
import os

import torch.utils.data as data
from PIL import Image


class Cityscapes(data.Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
        target_type (string, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, split='train', target_type='instance', transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, 'gtFine', split)
        self.transform = transform
        self.target_transform = target_transform
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid value for "split"! Please use split="train", split="train"'
                             ' or split="train"')

        if target_type not in ['instance', 'semantic', 'polygon', 'color']:
            raise ValueError('Invalid value for "target_type"! Please use target_type="instance",'
                             ' target_type="semantic", target_type="polygon" or target_type="color"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found. Please make sure both "leftImg8bit" and "gtFine"'
                               ' exist inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            lbl_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                label_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                            self._get_target_suffix(self.target_type))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(lbl_dir, label_name))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a json object if target_type="polygon",
            otherwise the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        if self.target_type == 'polygon':
            target = self._load_json(self.targets[index])
        else:
            target = Image.open(self.targets[index])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Type: {}\n'.format(self.target_type)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, target_type):
        if target_type == 'instance':
            return 'gtFine_instanceIds.png'
        elif target_type == 'semantic':
            return 'gtFine_labelIds.png'
        elif target_type == 'color':
            return 'gtFine_color.png'
        else:
            return 'gtFine_polygons.json'
