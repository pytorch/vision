import os
import os.path
from typing import Callable, Optional
from urllib.error import URLError

import numpy as np
import torch
from torchvision.datasets.utils import download_url
from torchvision.datasets.vision import VisionDataset


class MovingMNIST(VisionDataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MovingMNIST/raw/mnist_test_seq.npy`` exists.
        transform (callable, optional): A function/transform that takes in an torch Tensor
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    md5 = "be083ec986bfe91a449d63653c411eb2"

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data = self._load_data()

    def _load_data(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Batch of video frames (torch Tensor[B, C, T, H, W]).
                            The `B` is a batch size and `T` is the number of frames.
        """
        data = torch.from_numpy(
            np.load(os.path.join(self.raw_folder, self.raw_filename))
        )  # data has the shape of (sequence length, batch_size, height, width)
        data = torch.swapaxes(data, 0, 1)
        batch_size, seq_len, height, width = data.size()
        return torch.reshape(data, (batch_size, 1, seq_len, height, width))

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            index (int): Index
        Returns:
            torch.Tensor: Video frames (torch Tensor[C, T, H, W]). The `T` is the number of frames.
        """
        data = self.data[idx]
        if self.transform is not None:
            data = self._transform_sequences(data, self.transform)

        return data

    def _transform_sequences(self, frames: torch.Tensor, transform: Callable) -> torch.Tensor:
        """
        Args:
            frames (torch.Tensor): Tensor[C, T, H, W]. The `T` is the number of frames.
            transform (Calllable): A transform function.
        Returns:
            torch.Tensor: Transformed Tensor[C, T, H, W]. The `T` is the number of frames.
        """
        for frame_idx in range(frames.size(1)):
            frames[:, frame_idx] = transform(frames[:, frame_idx])
        return frames

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, self.raw_filename))

    def download(self) -> None:
        """Download the MovingMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        # Download file
        try:
            download_url(
                self.url,
                root=self.raw_folder,
                filename=self.raw_filename,
                md5=self.md5,
            )
        except URLError as error:
            print(f"Failed to download:\n{error}")
        finally:
            print()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def raw_filename(self) -> str:
        return self.url.split("/")[-1]
