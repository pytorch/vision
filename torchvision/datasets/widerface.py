from .vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from .utils import download_file_from_google_drive, download_and_extract_archive, check_integrity


class WIDERFace(VisionDataset):
    """`WIDERFace <http://shuoyang1213.me/WIDERFACE/>`_ Dataset.

    WIDER FACE dataset is a face detection benchmark dataset, of which images are 
    selected from the publicly available WIDER dataset. We choose 32,203 images and 
    label 393,703 faces with a high degree of variability in scale, pose and 
    occlusion as depicted in the sample images. WIDER FACE dataset is organized 
    based on 61 event classes. For each event class, we randomly select 40%/10%/50% 
    data as training, validation and testing sets. We adopt the same evaluation 
    metric employed in the PASCAL VOC dataset. Similar to MALF and Caltech datasets,
    we do not release bounding box ground truth for the test images. Users are 
    required to submit final prediction files, which we shall proceed to evaluate.

    @inproceedings{yang2016wider,
	    Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    Title = {WIDER FACE: A Face Detection Benchmark},
	    Year = {2016}}

    Args:
        root (string): Root directory of dataset where ``widerface/WIDER_train.zip widerface/WIDER_val.zip``
            and  ``widerface/WIDER_test.zip widerface/wider_face_split.zip`` exist.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            The specified dataset is selected.
            Defaults to ``train``.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "widerface"
    file_list = [
        # File ID                        MD5 Hash                            Filename
        ("0B6eKvaijfFUDQUUwd21EckhUbWs", "3fedf70df600953d25982bcd13d91ba2", "WIDER_train.zip"),
        ("0B6eKvaijfFUDd3dIRmpvSk8tLUk", "dfa7d7e790efa35df3788964cf0bbaea", "WIDER_val.zip"),
        ("0B6eKvaijfFUDbW4tdGpaYjgzZkU", "e5d8f4248ed24c334bbd12f49c29dd40", "WIDER_test.zip")
    ]
    annotations_file = ("http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip",
                        "0e3767bcf0e326556d407bf5bff5d27c",
                        "wider_face_split.zip")

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(WIDERFace, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        print("root dir: " + self.root)
        self.imgs_path = []
        self.words = []
        self.split = split

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it")

        print("Finished initializing WIDERFace")

        ann_file = os.path.expanduser(os.path.join(self.root, self.base_folder, "wider_face_split", "wider_face_train_bbx_gt.txt"))
        print("ann_file: " + ann_file)
        f = open(ann_file, "r")
        lines = f.readlines()

        isFile = True
        isNumBoxes, isBoxAnnotation = False, False
        num_boxes = 0
        box_counter = 0
        labels = []
        for line in lines:
            line = line.rstrip()
            if isFile:
                # print(line)
                self.imgs_path.append(line)
                isFile = False
                isNumBoxes = True
            elif isNumBoxes:
                num_boxes = int(line)
                isNumBoxes = False
                isBoxAnnotation = True
            elif isBoxAnnotation:
                box_counter += 1
                # line = line.split(" ")
                # line = [int(x) for x in line]
                # labels.append(line)
                if box_counter == num_boxes:
                    isBoxAnnotation = False
                    isFile = True
                    # print("read {} bounding boxes".format(box_counter))
                    # self.words.append(labels.copy())
                    box_counter = 0
                    # labels.clear()
            else:
                print("ERROR parsing annotations file")

        # isFirst = True
        # labels = []
        # for line in lines:
        #     line = line.rstrip()
        #     if line.startswith("#"):
        #         if isFirst is True:
        #             isFirst = False
        #         else:
        #             labels_copy = labels.copy()
        #             self.words.append(labels_copy)
        #             labels.clear()
        #         path = line[2:]
        #         path = ann_file.replace("label.txt","images/") + path
        #         self.imgs_path.append(path)
        #     else:
        #         line = line.split(" ")
        #         label = [float(x) for x in line]
        #         labels.append(label)
        # self.words.append(labels)


        # if self.train:
        #     data_file = self.training_file
        # else:
        #     data_file = self.test_file
        # self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target
        return 0, 1

    def __len__(self) -> int:
        return len(self.imgs_path)

    # TODO - checking integrity of the annotations_file is not working
    def _check_integrity(self) -> bool:
        all_files = self.file_list.copy()
        all_files.append(self.annotations_file)

        for (_, md5, filename) in all_files:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            print("filename: " + fpath)
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "WIDER_train"))

    def download(self) -> None:
        import zipfile

        # if self._check_integrity():
        #     print('Files already downloaded and verified')
        #     return

        # download data
        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        # extract data
        for (file_id, md5, filename) in self.file_list:
            with zipfile.ZipFile(os.path.join(self.root, self.base_folder, filename), "r") as f:
                f.extractall(os.path.join(self.root, self.base_folder))

        # download and extract annotations files
        download_and_extract_archive(url=self.annotations_file[0],
                                     download_root=os.path.join(self.root, self.base_folder),
                                     extract_root=os.path.join(self.root, self.base_folder),
                                     filename=self.annotations_file[2],
                                     md5=self.annotations_file[1])
