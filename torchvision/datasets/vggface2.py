from functools import partial
from PIL import Image
import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .utils import check_integrity, extract_archive
from .vision import VisionDataset


class VGGFace2(VisionDataset):
    """ VGGFace2 <http://zeus.robots.ox.ac.uk/vgg_face2/>`_ Dataset.

        Args:
            root (string): Root directory of the VGGFace2 Dataset.
            Expects the following folder structure if download=False:
                .
                └── vggface2
                    ├── vggface2_train.tar.gz (or 'train' if uncompressed)
                    ├── vggface2_test.tar.gz (or 'test' if uncompressed)
                    ├── train_list.txt
                    ├── test_list.txt
                    └── bb_landmark.tar.gz (or 'bb_landmark' if uncompressed)
            split (string): One of {``train``, ``test``}.
                The dataset split to use. Defaults to ``train``.
            target_type (string): The type of target to use. One of
                {``class_id``, ``image_id``, ``face_id``, ``bbox``, ``landmarks``.``""``}
                Can also be a list to output a tuple with all specified target types.
                The targets represent:
                    ``class_id`` (string)
                    ``image_id`` (string)
                    ``face_id`` (string)
                    ``bbox`` (torch.tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                    ``landmarks`` (torch.tensor shape=(10,) dtype=float): values that
                        represent five points (P1X, P1Y, P2X, P2Y, P3X, P3Y, P4X, P4Y, P5X, P5Y)
                Defaults to ``bbox``. If empty, ``None`` will be returned as target.
            transform (callable, optional): A function/transform that  takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """

    base_folder = "vggface2"
    file_list = [
        # Filename                MD5 Hash                            Uncompressed filename
        ("vggface2_train.tar.gz", "88813c6b15de58afc8fa75ea83361d7f", "train"),
        ("vggface2_test.tar.gz", "bb7a323824d1004e14e00c23974facd3", "test"),
        ("bb_landmark.tar.gz", "26f7ba288a782862d137348a1cb97540", "bb_landmark")
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "bbox",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        import pandas
        super(VGGFace2, self).__init__(root=os.path.join(root, self.base_folder),
                                       transform=transform,
                                       target_transform=target_transform)

        # stay consistent with other datasets and check for a download option
        if download:
            msg = ("The dataset is not publicly accessible. You must login and "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)

        # check arguments
        if split not in ('train', 'test'):
            raise ValueError('split \"{}\" is not recognized.'.format(split))
        self.split = split
        self.img_info: List[Dict[str, object]] = []

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not (all(x in ["class_id", "image_id", "face_id", "bbox", "landmarks", ""] for x in self.target_type)):
            raise ValueError("target_type \"{}\" is not recognized.".format(self.target_type))
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        image_list_file = 'train_list.txt' if self.split == 'train' else 'test_list.txt'
        self.image_list_file = os.path.join(self.root, image_list_file)

        # prepare dataset
        for (filename, _, extracted_dir) in self.file_list:
            filename = os.path.join(self.root, filename)
            extracted_dir_path = os.path.join(self.root, extracted_dir)
            if not os.path.isdir(extracted_dir_path):
                extract_archive(filename)

        # process dataset
        fn = partial(os.path.join, self.root, self.file_list[2][2])
        bbox_frames = [pandas.read_csv(fn("loose_bb_train.csv"), index_col=0),
                       pandas.read_csv(fn("loose_bb_test.csv"), index_col=0)]
        self.bbox = pandas.concat(bbox_frames)
        landmark_frames = [pandas.read_csv(fn("loose_landmark_train.csv"), index_col=0),
                           pandas.read_csv(fn("loose_landmark_test.csv"), index_col=0)]
        self.landmarks = pandas.concat(landmark_frames)

        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()
                img_filename, ext = os.path.splitext(img_file)  # e.g. ["n004332/0317_01", "jpg"]
                class_id, image_face_id = img_filename.split("/")
                image_id, face_id = image_face_id.split("_")
                img_filepath = os.path.join(self.root, self.split, img_file)
                self.img_info.append({
                    'img_path': img_filepath,
                    'class_id': class_id,
                    'image_id': image_id,
                    'face_id': face_id,
                    'bbox': torch.tensor(self.bbox.loc[img_filename].values),
                    'landmarks': torch.tensor(self.landmarks.loc[img_filename].values),
                })

    def __len__(self) -> int:
        return len(self.img_info)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        img_info = self.img_info[index]

        # prepare image
        img = Image.open(img_info['img_path'])
        if self.transform:
            img = self.transform(img)

        # prepare target
        target: Any = []
        for t in self.target_type:
            if t == "":
                target = None
                break
            target.append(img_info[t])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
