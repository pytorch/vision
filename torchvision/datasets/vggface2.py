from PIL import Image
import os
import torch
from .vision import VisionDataset
from .utils import check_integrity, extract_archive

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
                    └── test_list.txt
            split (string): One of {``train``, ``test``}.
                The dataset split to use. Defaults to ``train``.
            target_type (string): The type of target to use, can be one of {``identity``, ``bbox``, ``attr``.``""``}
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
        """
    
    base_folder = "vggface2"
    file_list = [
        # Filename                MD5 Hash                            Uncompressed filename
        ("vggface2_train.tar.gz", "88813c6b15de58afc8fa75ea83361d7f", "train"),
        ("vggface2_test.tar.gz",  "bb7a323824d1004e14e00c23974facd3", "test"),
    ]

    def __init__(
        self,
        root,
        split = "train",
        download = False,
        transform = None,
        target_transform = None):

        root = os.path.join(root, self.base_folder)
        super(VGGFace2, self).__init__(root, transform=transform, target_transform=target_transform)

        if download:
            msg = ("The dataset is not publicly accessible. You must login and "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)

        # check arguments
        if split not in ('train', 'test'):
            raise ValueError('split \"{}\" is not recognized.'.format(split))
        self.split = split
        self.img_info = []

        image_list_file = 'train_list.txt' if self.split == 'train' else 'test_list.txt'
        self.image_list_file = os.path.join(self.root, image_list_file)

        # are files downloaded and extracted?
        for (filename, md5, extracted_folder) in self.file_list:
            extracted_folder_path = os.path.join(self.root, extracted_folder)
            if not os.path.isdir(extracted_folder_path):
                raise RuntimeError('Can not find folder \"{}\"'.format(extracted_folder_path))

        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()        # e.g. n004332/0317_01.jpg
                class_id = img_file.split("/")[0]  # like n004332
                img_file = os.path.join(self.root, self.split, img_file) # like root/vggface2/train/n000002/0001_01.jpg
                print("img_file: " + img_file)
                self.img_info.append({
                    'img_path': img_file,
                    'class_id': class_id,
                })

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        img_info = self.img_info[index]
        img = Image.open(img_info['img_path'])
        if self.transform:
            img = self.transform(img)

        target = None
        if self.split == "test":
            return img, target

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def extra_repr(self) -> str:
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
