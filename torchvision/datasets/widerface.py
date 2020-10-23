from PIL import Image
import os
import torch
from typing import Any, Callable, List, Optional, Tuple, Union
from .utils import download_file_from_google_drive, download_and_extract_archive, check_integrity
from .vision import VisionDataset


class WIDERFace(VisionDataset):
    """`WIDERFace <http://shuoyang1213.me/WIDERFACE/>`_ Dataset.

    Citation:
    @inproceedings{yang2016wider,
	    Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    Title = {WIDER FACE: A Face Detection Benchmark},
	    Year = {2016}}

    Args:
        root (string): Root directory of the WIDERFace Dataset.
            Expects the following structure:
                .
                └── widerface
                    ├── wider_face_split.zip
                    ├── WIDER_test.zip
                    ├── WIDER_train.zip
                    └── WIDER_val.zip
        split (string): One of {``train``, ``val``, ``test``}.
            The dataset split to use. Defaults to ``train``.
        target_type (string): The type of target to use, can be one of {``raw``, ``bbox``, ``attr``.``""``}
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``raw`` (torch.tensor shape=(10,) dtype=int): all annotations combined (bbox + attr)
                ``bbox`` (torch.tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``attr`` (torch.tensor shape=(6,) dtype=int): label values for attributes
                    that represent (blur, expression, illumination, occlusion, pose, invalid)
            Defaults to ``raw``. If empty, ``None`` will be returned as target.
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
            target_type: Union[List[str], str] = "raw",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(WIDERFace, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        # check arguments
        if split not in ("train","val","test"):
            raise ValueError("split \"{}\" is not recognized.".format(split))
        self.split = split

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        
        if not (all(x in ["raw","bbox","attr",""] for x in self.target_type)):
            raise ValueError("target_type \"{}\" is not recognized.".format(self.target_type))
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        # prepare dataset
        self.imgs_path = []
        self.raw_annotations = []

        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it")

        # process dataset
        if self.split in ("train","val"):
            self.parse_train_val_annotations_file()
        elif self.split == "test":
            self.parse_test_annotations_file()
        else:
            raise ValueError("split \"{}\" is not recognized.".format(self.split))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target=None for the test split.
        """

        # stay consistent with other datasets and return a PIL Image
        img = Image.open(self.imgs_path[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.split == "test":
            return img, None

        # prepare target in the train/val split
        target: Any = []
        for t in self.target_type:
            if t == "raw":
                target.append( self.raw_annotations[index] )
            elif t == "bbox":
                target.append( self.raw_annotations[index][:,:4] )
            elif t == "attr":
                target.append( self.raw_annotations[index][:,4:] )
            elif t == "":
                target = None
                break
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs_path)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def parse_train_val_annotations_file(self) -> None:
        filename = "wider_face_train_bbx_gt.txt" if self.split == "train" else "wider_face_val_bbx_gt.txt"
        filepath = os.path.join(self.root, self.base_folder, "wider_face_split", filename)

        f = open(filepath, "r")
        lines = f.readlines()

        file_name_line, num_boxes_line, box_annotation_line = True, False, False
        num_boxes, box_counter = 0, 0
        labels = []
        for line in lines:
            line = line.rstrip()
            if file_name_line:
                abs_path = os.path.join(self.root, self.base_folder, "WIDER_"+self.split, "images", line)
                self.imgs_path.append(abs_path)
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line = line.split(" ")
                line = [int(x) for x in line]
                labels.append(line)
                if box_counter >= num_boxes:
                    box_annotation_line = False
                    file_name_line = True
                    self.raw_annotations.append(torch.tensor(labels))
                    box_counter = 0
                    labels.clear()
            else:
                raise RuntimeError("Error parsing annotation file {}".format(filepath))
        f.close()

    def parse_test_annotations_file(self) -> None:
        filepath = os.path.join(self.root, self.base_folder, "wider_face_split", "wider_face_test_filelist.txt")
        f = open(filepath, "r")
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            abs_path = os.path.join(self.root, self.base_folder, "WIDER_test", "images", line)
            self.imgs_path.append(abs_path)
        f.close()

    def _check_integrity(self) -> bool:
        all_files = self.file_list.copy()
        all_files.append(self.annotations_file)
        for (_, md5, filename) in all_files:
            fpath = os.path.join(self.root, self.base_folder, filename)
            file, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip). Only need the extracted images
            # Should check a hash of the images
            extracted_dir = os.path.join(self.root, self.base_folder, file)
            if ext != ".zip" and not check_integrity(fpath, md5):
                return False
            if not os.path.isdir(extracted_dir):
                return False
        return True

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download data
        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id,
                                            os.path.join(self.root, self.base_folder),
                                            filename, md5)

        # extract data if it doesn't exist
        for (file_id, md5, filename) in self.file_list:
            file, _ = os.path.splitext(filename)
            extracted_dir = os.path.join(self.root, self.base_folder, file)
            if not os.path.isdir(extracted_dir):
                zip_file = os.path.join(self.root, self.base_folder, filename)
                with zipfile.ZipFile(zip_file, "r") as f:
                    new_extracted_dir = os.path.join(self.root, self.base_folder)
                    f.extractall(new_extracted_dir)

        # download and extract annotation files
        download_and_extract_archive(url=self.annotations_file[0],
                                     download_root=os.path.join(self.root, self.base_folder),
                                     extract_root=os.path.join(self.root, self.base_folder),
                                     filename=self.annotations_file[2],
                                     md5=self.annotations_file[1])
