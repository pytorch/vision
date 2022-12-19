import os
import os.path
from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets.utils import download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class MovingMNIST(VisionDataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MovingMNIST/raw/mnist_test_seq.npy`` exists.
        split (string, optional): The dataset split, supports ``None`` (default), ``"train"`` and ``"test"``.
            If ``split=None``, the full data is returned.
        split_ratio (int, optional): The split ratio of datasets. If ``split="train"``, the first split
            ``data[:split_ratio]`` is returned. If ``split="test"``, the last split ``data[split_ratio:]``
            is returned. If ``split=None``, this parameter is ignored and the full data is returned.
        transform (callable, optional): A function/transform that takes in an torch Tensor
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        split_ratio: int = 10,
        download: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)

        if split is not None:
            verify_str_arg(split, "split", ("train", "test"))
        self.split = split
        self.split_ratio = split_ratio

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data = self._load_data()

    def _load_data(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Batch of video frames (torch Tensor[B, T, C, H, W]).
                            The `B` is a batch size and `T` is the number of frames.
        """
        data = torch.from_numpy(
            np.load(os.path.join(self.raw_folder, self.raw_filename))
        )  # data has the shape of (sequence length, batch_size, height, width)
        data = torch.swapaxes(data, 0, 1)
        batch_size, seq_len, height, width = data.size()
        data = torch.reshape(data, (batch_size, seq_len, 1, height, width))
        if self.split is None:
            return data
        elif self.split == "train":
            return data[: self.split_ratio]
        else:
            return data[self.split_ratio :]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            index (int): Index
        Returns:
            torch.Tensor: Video frames (torch Tensor[T, C, H, W]). The `T` is the number of frames.
        """
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, self.raw_filename))

    def download(self) -> None:
        """Download the MovingMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        download_url(
            url="http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
            root=self.raw_folder,
            filename=self.raw_filename,
            md5="be083ec986bfe91a449d63653c411eb2",
        )

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def raw_filename(self) -> str:
        return "mnist_test_seq.npy"
