import torch
import torchvision
from PIL import Image
from torchvision import transforms as T

import os
import numpy as np
import random

from pycocotools import mask as coco_mask


class VOC(torchvision.datasets.VOCSegmentation):
    def __init__(self, root, image_set='train', transforms=None):
        super(VOC, self).__init__(root, image_set=image_set)
        self.transforms = transforms

    def __getitem__(self, idx):
        image, target = super(VOC, self).__getitem__(idx)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    @property
    def num_classes(self):
        return 21

class SBDD(object):
    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        with open(os.path.join(root, image_set + '.txt')) as f:
            self.ids = f.readlines()
        self.ids = [x.strip() for x in self.ids]
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = Image.open(os.path.join(self.root, 'img', img_id + '.jpg')).convert('RGB')
        target = Image.open(os.path.join(self.root, 'cls', img_id + '.png'))
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.ids)

    @property
    def num_classes(self):
        return 21


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if more than 1k pixels occupied in the image
    return sum(obj["area"] for obj in anno) > 1000


def convert_polys_to_mask(anno, h, w, categories):
    target = torch.zeros((h, w), dtype=torch.uint8)
    anno = [obj for obj in anno if obj["category_id"] in categories]
    for instance in anno:
        rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
        m = coco_mask.decode(rle)
        cat = instance['category_id']
        c = categories.index(cat)
        m = torch.as_tensor(m)
        if len(m.shape) < 3:
            m = m[..., None]
        target = torch.where(target == 0, m.any(dim=2) * c, target)
    return target


class _COCO(torchvision.datasets.CocoDetection):

    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]
    NUM_CLASS = 21
    def __init__(self, root, ann_file, remove_images_without_annotations, transforms=None):
        super(_COCO, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                anno = [obj for obj in anno if obj["category_id"] in self.CAT_LIST]
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.transforms = transforms

    def __getitem__(self, idx):
        image, anno = super(_COCO, self).__getitem__(idx)

        w, h = image.size
        target = convert_polys_to_mask(anno, h, w, self.CAT_LIST)
        target = Image.fromarray(target.numpy())

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    @property
    def num_classes(self):
        return 21

class COCO(torch.utils.data.ConcatDataset):
    PATHS = {
        "train": [("train2014", os.path.join("annotations", "instances_train2014.json")),
                ("val2014", os.path.join("annotations", "instances_valminusminival2014.json"))],
        "val": [("val2014", os.path.join("annotations", "instances_minival2014.json"))]
    }
    def __init__(self, root, image_set, transforms):
        p = self.PATHS[image_set]
        remove_images_without_annotations = True  # TODO use train / val to decide
        datasets = []
        for img_folder, ann_file in p:
            img_folder = os.path.join(root, img_folder)
            ann_file = os.path.join(root, ann_file)
            datasets.append(_COCO(img_folder, ann_file, remove_images_without_annotations, transforms))
        super(COCO, self).__init__(datasets)

    @property
    def num_classes(self):
        return 21

def pad_if_smaller(img, size):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = T.functional.pad(img, (0, 0, padw, padh))
    return img


class Transform(object):
    def __init__(self, flip_prob=0):
        self.base_size = 520
        self.crop_size = 480
        self.flip_prob = flip_prob
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, image, target):
        if self.flip_prob != 0:  # TODO fix
            size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        else:
            size = self.base_size
        image = T.functional.resize(image, size)
        target = T.functional.resize(target, size, interpolation=Image.NEAREST)
        if random.random() < self.flip_prob:
            image = T.functional.hflip(image)
            target = T.functional.hflip(target)

        if self.flip_prob != 0:  # TODO fix
            cs = self.crop_size
            image = pad_if_smaller(image, cs)
            target = pad_if_smaller(target, cs)
            crop_params = T.RandomCrop.get_params(image, (cs, cs))
            image = T.functional.crop(image, *crop_params)
            target = T.functional.crop(target, *crop_params)
        else:
            cs = self.crop_size
            image = T.functional.center_crop(image, cs)
            target = T.functional.center_crop(target, cs)

        image = self.to_tensor(image)
        target = torch.as_tensor(np.asarray(target)).to(torch.int64)
        image = self.normalize(image)

        return image, target
