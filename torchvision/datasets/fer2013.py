import csv
import pathlib
import warnings
from itertools import islice
from typing import Any, Callable, Optional, Tuple, Union

warnings.simplefilter("always", Warning)

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
    }

    _RESOURCES_EXTRA = {"fer2013": ("fer2013.csv", "f8428a1edbd21e88f42c73edd2a14f95")}

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._split = verify_str_arg(split, "split", self._RESOURCES.keys())
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root) / "fer2013"
        file_name, md5 = self._RESOURCES[self._split]
        data_file = base_folder / file_name
        if not check_integrity(str(data_file), md5=md5):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        with open(data_file, "r", newline="") as file:
            self._samples = [
                (
                    torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                    int(row["emotion"]) if "emotion" in row else None,
                )
                for row in csv.DictReader(file)
            ]

        self.get_test_label = False
        self.err_fer2013_warning = (
            "The integrity of fer2013.csv has been compromised, "
            "thus the test labels will not be provided. "
            "You can download it from "
            "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
        )
        if self._split == "test":
            self.get_test_label = True

        if self.get_test_label:
            try:
                file_name_entire, md5_entire = self._RESOURCES_EXTRA["fer2013"]
                entire_data_file = base_folder / file_name_entire
                if not check_integrity(str(entire_data_file), md5=md5_entire):
                    self.get_test_label = False
                    warnings.warn(self.err_fer2013_warning, Warning)
            except KeyError:
                self.get_test_label = False
                warnings.warn(self.err_fer2013_warning, Warning)

        if self.get_test_label:
            with open(entire_data_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                test_rows = islice(reader, 28709, None)
                self._samples_alt = [
                    (
                        torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                        int(row["emotion"]) if "emotion" in row else None,
                    )
                    for row in test_rows
                ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]

        if self.get_test_label:
            _, target_alt = self._samples_alt[idx]
            target = target_alt

        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"
