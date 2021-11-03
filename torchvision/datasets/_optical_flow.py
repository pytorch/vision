import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..io.image import _read_png_16
from .vision import VisionDataset


__all__ = (
    "KittiFlow",
    "Sintel",
)


class FlowDataset(ABC, VisionDataset):
    def __init__(self, root, transforms=None):

        super().__init__(root=root)
        self.transforms = transforms

        self._flow_list = []
        self._image_list = []

    def _read_img(self, file_name):
        return Image.open(file_name)

    @abstractmethod
    def _read_flow(self, file_name):
        # Return the flow or a tuple (flow, valid) for datasets where the valid mask is built-in
        pass

    def __getitem__(self, index):
        # Some datasets like Kitti have a built-in valid mask, indicating which flow values are valid
        # For those we return (img1, img2, flow, valid), and for the rest we return (img1, img2, flow),
        # and it's up to whatever consumes the dataset to decide what `valid` should be.

        img1 = self._read_img(self._image_list[index][0])
        img2 = self._read_img(self._image_list[index][1])
        flow = self._read_flow(self._flow_list[index]) if self._flow_list else None

        if isinstance(flow, tuple):
            flow, valid = flow
        else:
            valid = None

        if self.transforms is not None:
            img1, img2, flow, valid = self.transforms(img1, img2, flow, valid)

        if valid is None:
            return img1, img2, flow
        else:
            return img1, img2, flow, valid

    def __len__(self):
        return len(self._image_list)


class Sintel(FlowDataset):
    def __init__(
        self,
        root,
        split="train",
        dstype="clean",
        transforms=None,
    ):

        super().__init__(root=root, transforms=transforms)

        if split not in ("train", "test"):
            raise ValueError("split must be either 'train' or 'test'")

        if dstype not in ("clean", "final"):
            raise ValueError("dstype must be either 'clean' or 'final'")

        split_dir = "training" if split == "train" else split
        flow_root = Path(root) / split_dir / "flow"
        image_root = Path(root) / split_dir / dstype

        for scene in os.listdir(image_root):
            image_list = sorted(glob(str(image_root / scene / "*.png")))
            for i in range(len(image_list) - 1):
                self._image_list += [[image_list[i], image_list[i + 1]]]

            if split == "train":
                self._flow_list += sorted(glob(str(flow_root / scene / "*.flo")))

    def _read_flow(self, file_name):
        return _read_flo(file_name)


class KittiFlow(FlowDataset):
    def __init__(
        self,
        root,
        split="train",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ("train", "test"):
            raise ValueError("split must be either 'train' or 'test'")

        root = Path(root) / ("training" if split == "train" else split)
        images1 = sorted(glob(str(root / "image_2" / "*_10.png")))
        images2 = sorted(glob(str(root / "image_2" / "*_11.png")))

        for img1, img2 in zip(images1, images2):
            self._image_list += [[img1, img2]]

        if split == "train":
            self._flow_list = sorted(glob(str(root / "flow_occ" / "*_10.png")))

    def _read_flow(self, file_name):
        return _read_16bits_png_with_flow_and_valid_mask(file_name)


def _read_flo(file_name):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise ValueError("Magic number incorrect. Invalid .flo file")

        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print 'Reading %d x %d flo file\n' % (w, h)
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # Reshape data into 3D array (columns, rows, bands)
        # The reshape here is for visualization, the original code is (w,h,2)
        return np.resize(data, (int(h), int(w), 2))


def _read_16bits_png_with_flow_and_valid_mask(file_name):

    flow_and_valid = _read_png_16(file_name).to(torch.float32)
    flow, valid = flow_and_valid[:2, :, :], flow_and_valid[2, :, :]
    flow = (flow - 2 ** 15) / 64  # This conversion is explained somewhere on the kitti archive

    return flow, valid
