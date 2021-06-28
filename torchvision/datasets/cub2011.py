from collections import namedtuple
import csv
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity, verify_str_arg, extract_archive
import numpy as np


class Cub2011(VisionDataset):
    """`CUB <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional):List of target to use, ``class_label``, ``segmentation`` or ``bbox``.
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                - ``class_label`` (int): range (0-200) labels for attributes
                - ``segmentation`` (float): segmentation map of each input Image
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
            Defaults to ``class_label``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "Cub2011"
    file_list = [
        # File ID                             MD5 Hash                            Filename
        ("1hbzc_P1FuxMkcabkgn9ZKinBwW683j45", "97eceeb196236b17998738112f37df78", "CUB_200_2011.tgz"),
        ("1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP", "4d47ba1228eae64f2fa547c47bc65255", "segmentations.tgz"),
    ]

    meta_data = {"image_lst": "images.txt", "class_labels": "image_class_labels.txt",
                 "split_lst": "train_test_split.txt", "bb_lst": "bounding_boxes.txt"}

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = ["class_label"],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.target_type = target_type

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.root = root
        split_map = {
            "test": 0,
            "train": 1,
            "all": 2,
        }

        self._meta_data = {}
        for key in self.meta_data.keys():
            self._meta_data[key] = self.filter_data(key)

        self.index_list = self.process_indexs(self._meta_data['split_lst'], split_map[split])

        print('Dataset Loaded Successfully')

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)
            extract_archive(os.path.join(self.root, self.base_folder, filename))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename = self._meta_data["image_lst"][self.index_list[index]]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "CUB_200_2011", 'images', filename)).convert('RGB')

        target: Any = []
        for t in self.target_type:
            if t == "class_label":
                target.append(torch.tensor(float(self._meta_data["class_labels"][index])))
            elif t == "segmentation":
                segmentaiton = PIL.Image.open(os.path.join(self.root, self.base_folder, "segmentations",
                                                           filename.replace('.jpg', '.png'))).convert('1')
                target.append(torch.tensor(np.asarray(segmentaiton), dtype=torch.float32))
            elif t == "bbox":
                target.append(torch.tensor(np.asarray(self._meta_data["bb_lst"][index]).astype(np.float)))
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.index_list)

    def process_indexs(self, list, target):
        processed_index = []
        for i in range(len(list)):
            if target == 2:
                processed_index.append(i)
            elif int(list[i]) == target:
                processed_index.append(i)
        return processed_index

    def filter_data(self, key):
        filter_data_lst = []
        for ind, data in enumerate(open(os.path.join(self.root, self.base_folder, "CUB_200_2011",
                                                     self.meta_data[key]), 'r').readlines()):
            data = data[:-1].split(' ')
            if len(data) == 2:
                filter_data_lst.append(data[1])
            else:
                filter_data_lst.append(data[1:])
        return filter_data_lst
