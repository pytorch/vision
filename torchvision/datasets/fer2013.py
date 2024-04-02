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
        "train": ("icml_face_data.csv", "b114b9e04e6949e5fe8b6a98b3892b1d"),
        "test": ("icml_face_data.csv", "b114b9e04e6949e5fe8b6a98b3892b1d"),
    }

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
            reader = csv.DictReader(file)
            self._samples = []
            for row in reader:
                cleaned_row = {name.strip(): value for name, value in row.items()}
                if self._split in cleaned_row["Usage"].lower():
                    self._samples.append(
                        (
                            torch.tensor(
                                [int(idx) for idx in cleaned_row["pixels"].split()], dtype=torch.uint8
                            ).reshape(48, 48),
                            int(cleaned_row["emotion"]) if "emotion" in cleaned_row else None,
                        )
                    )

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
