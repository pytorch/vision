import csv
import pathlib
from typing import Any, Callable, Optional, Tuple, Union

import torch
from PIL import Image

from .utils import check_integrity, verify_str_arg
from .vision import VisionDataset


class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _RESOURCES = {
        "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
        # This one also contains both train and tests instances, and unlike test.csv it contains the labels
        # for the test instances.
        # It is used if it exists, otherwise "train" and "test" are used for BC, as support for "icml" was added later.
        "icml": ("icml_face_data.csv", "b114b9e04e6949e5fe8b6a98b3892b1d"),
    }

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "test"))
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root) / "fer2013"
        use_icml = (base_folder / self._RESOURCES["icml"][0]).exists()
        file_name, md5 = self._RESOURCES["all" if use_icml else self._split]
        data_file = base_folder / file_name
        if not check_integrity(str(data_file), md5=md5):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        pixels_key = " pixels" if use_icml else "pixels"  # yes, for real

        def get_img(row):
            return torch.tensor([int(idx) for idx in row[pixels_key].split()], dtype=torch.uint8).reshape(48, 48)

        def get_label(row):
            if use_icml or self._split == "train":
                return int(row["emotion"])
            else:
                return None

        with open(data_file, "r", newline="") as file:
            rows = (row for row in csv.DictReader(file))

            if use_icml:
                valid_keys = ("Training",) if self._split == "train" else ("PublicTest", "PrivateTest")
                rows = (row for row in rows if row[" Usage"] in valid_keys)

            self._samples = [(get_img(row), get_label(row)) for row in rows]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"
