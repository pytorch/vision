import os
from pathlib import Path
from typing import Any, Tuple, Callable, Optional, Union

import PIL.Image

from .utils import verify_str_arg, download_and_extract_archive, check_integrity
from .vision import VisionDataset


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.
    The SUN397 is a dataset for scene recognition consisting of 397 categories with 108'754 images.
    The dataset also provides 10 paritions for training and testing, with each partition 
    consisting of 50 images per class. 
    
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        parition (string, integer, optional): A valid partition can be an integer from 1 to 10 or ``"all"``
            for the entire dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _FILENAME = "SUN397.tar.gz"
    _MD5 = "8ca2778205c41d23104230ba66911c7a"
    _PARTITIONS_URL = "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
    _PARTITIONS_FILENAME = "Partitions.zip"

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: Union[int,str] = 1,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.partition = partition
        self.data_dir = Path(self.root) / "SUN397"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self._labels = []
        self._image_files = []
        with open(self.data_dir / f"ClassName.txt", "r") as f:
            classes = f.read().splitlines()
            
        for idx,c in enumerate(classes):
            classes[idx] = c[3:]

        self.classes = classes
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        
        if isinstance(self.partition,int):
            if self.partition<0 or self.partition>10:
                raise RuntimeError("Enter a valid integer partition from 1 to 10 or \"all\" ")
            
            splitname = "Training" if self.split is "train" else "Testing"
            zero = "0" if self.partition<10 else ""
            
            with open(self.data_dir / f"{splitname}_{zero}{self.partition}.txt", "r") as f:
                pathlist = f.read().splitlines()
                
            for p in pathlist:
                self._labels.append(self.class_to_idx[p[3:-25]])
                self._image_files.append(self.data_dir.joinpath(*p.split("/")))
            
        else:
            if self.partition is not "all":
                raise RuntimeError("Enter a valid integer partition from 1 to 10 or \"all\" ")
            else:
                for path, _, files in os.walk(self.data_dir):
                    for file in files:
                        if(file[:3]=="sun"):
                            self._image_files.append(Path(path)/file)
                            self._labels.append(Path(path).relative_to(self.data_dir).as_posix()[2:])
                

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        file = Path(self.root) / self._FILENAME
        if not check_integrity(file, self._MD5):
            return False
        elif self._PARTITIONS_FILENAME not in os.listdir(self.data_dir):
            return False
        else:
            return True
    
    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def _download(self) -> None:
        file = Path(self.root) / self._FILENAME
        if self._FILENAME not in os.listdir(self.data_dir) or not check_integrity(file, self._MD5):
            download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)
            
        if self._PARTITIONS_FILENAME not in os.listdir(self.data_dir):
            download_and_extract_archive(self._PARTITIONS_URL, download_root=self.data_dir)