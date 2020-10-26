import os
from PIL import Image
import torch
from torchvision.datasets import VisionDataset

class VGGFace2(VisionDataset):

    def __init__(self, root, split="train", transform=None, target_transform=None):
        """
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
        super(VGGFace2, self).__init__(root, transform=transform, target_transform=target_transform)
        # check arguments
        if split not in ('train','test'):
            raise ValueError('split \"{}\" is not recognized.'.format(split))
        self.split = split
        self.img_info = []

        image_list_file = 'train_list.txt' if self.split=='train' else 'test_list.txt'
        self.image_list_file = os.path.join(self.root, image_list_file)

        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()  # e.g. train/n004332/0317_01.jpg
                class_id = img_file.split("/")[0]  # like n004332
                img_file = os.path.join(self.root, self.split, img_file)
                self.img_info.append({
                    'img_path': img_file,
                    'class_id': class_id,
                })
                if i % 1000 == 0:
                    print("processing: {} images for {}".format(i, self.split))

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
