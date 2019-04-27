import copy
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T

import os
import numpy as np
import random

from pycocotools import mask as coco_mask


def convert_polys_to_mask(segmentations, categories, h, w):
    target = torch.zeros((h, w), dtype=torch.uint8)
    for segm, cat in zip(segmentations, categories):
        m = convert_coco_poly_to_mask(segm, h, w)
        target = torch.where(target == 0, m * cat, target)
        # can also cat and do a max over dim 2
    return target

def convert_coco_poly_to_mask(polygons, height, width):
    rles = coco_mask.frPyObjects(polygons, height, width)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
        mask = mask[..., None]
    mask = torch.as_tensor(mask)
    mask = mask.any(dim=2)
    return mask

class FilterAndRemapCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


class ConvertPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        target = convert_polys_to_mask(segmentations, cats, h, w)
        target = Image.fromarray(target.numpy())
        return image, target

def _coco_has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if more than 1k pixels occupied in the image
    return sum(obj["area"] for obj in anno) > 1000

def _coco_remove_images_without_annotations(dataset, cat_list=None):
    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for img_id in dataset.ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _coco_has_valid_annotation(anno):
            ids.append(img_id)
    dataset.ids = ids
    return dataset

def get_coco(root, image_set, transforms):
    PATHS = {
        #"train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    transforms = Compose([
        FilterAndRemapCategories(CAT_LIST, remap=True),
        ConvertPolysToMask(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)
    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    #if image_set == "train":
    #    _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = T.functional.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = T.functional.resize(image, size)
        target = T.functional.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = T.functional.hflip(image)
            target = T.functional.hflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = T.functional.crop(image, *crop_params)
        target = T.functional.crop(target, *crop_params)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = T.functional.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(RandomResize(min_size, max_size))
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomCrop(crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))

    return Compose(transforms)
