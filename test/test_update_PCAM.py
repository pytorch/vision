import bz2
import contextlib
import csv
import io
import itertools
import json
import os
import pathlib
import pickle
import random
import re
import shutil
import string
import unittest
import xml.etree.ElementTree as ET
import zipfile
from typing import Callable, Union

import datasets_utils
import numpy as np
import PIL
import pytest
import torch
import torch.nn.functional as F
from common_utils import combinations_grid
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import v2


class PCAMTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.PCAM

    ADDITIONAL_CONFIGS = combinations_grid(split=("train", "val", "test"))
    REQUIRED_PACKAGES = ("h5py",)

    def inject_fake_data(self, tmpdir: str, config):
        base_folder = pathlib.Path(tmpdir) / "pcam"
        base_folder.mkdir()

        num_images = {"train": 2, "test": 3, "val": 4}[config["split"]]

        images_file = datasets.PCAM._FILES[config["split"]]["images"][0]
        with datasets_utils.lazy_importer.h5py.File(str(base_folder / images_file), "w") as f:
            f["x"] = np.random.randint(0, 256, size=(num_images, 10, 10, 3), dtype=np.uint8)

        targets_file = datasets.PCAM._FILES[config["split"]]["targets"][0]
        with datasets_utils.lazy_importer.h5py.File(str(base_folder / targets_file), "w") as f:
            f["y"] = np.random.randint(0, 2, size=(num_images, 1, 1, 1), dtype=np.uint8)

        return num_images

if __name__ == "__main__":
    unittest.main()
