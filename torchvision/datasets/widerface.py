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
from .utils import download_and_extract_archive


class WIDERFace(VisionDataset):
    """`WIDERFace <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

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
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
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

    file_list = [
        # Download URL                                                                  MD5 Hash                            Filename
        # ("https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDQUUwd21EckhUbWs", "3fedf70df600953d25982bcd13d91ba2", "WIDER_train.zip"),
        ("https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk", "dfa7d7e790efa35df3788964cf0bbaea", "WIDER_val.zip"),
        # ("https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU", "e5d8f4248ed24c334bbd12f49c29dd40", "WIDER_test.zip")
    ]

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
        print("root dir: " + root)
        print(self.root)
        self.split = split

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        # if download:
        #     self.download()

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found.' +
        #                        ' You can use download=True to download it')
        
        print("done downloading wider face")

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
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    def download(self) -> None:
        for (file_url, md5, filename) in self.file_list:
            download_and_extract_archive(url=file_url, download_root=self.root, filename=filename, md5=md5)
            # download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        # with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
        #     f.extractall(os.path.join(self.root, self.base_folder))
    
    # def download_mnist(self) -> None:
    #     """Download the MNIST data if it doesn't exist in processed_folder already."""

    #     if self._check_exists():
    #         return

    #     os.makedirs(self.raw_folder, exist_ok=True)
    #     os.makedirs(self.processed_folder, exist_ok=True)

    #     # download files
    #     for url, md5 in self.resources:
    #         filename = url.rpartition('/')[2]
    #         download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

    #     # process and save as torch files
    #     print('Processing...')

    #     training_set = (
    #         read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
    #         read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
    #     )
    #     test_set = (
    #         read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
    #         read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
    #     )
    #     with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
    #         torch.save(training_set, f)
    #     with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
    #         torch.save(test_set, f)

    #     print('Done!')
