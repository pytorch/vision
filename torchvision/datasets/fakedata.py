from typing import Any, Callable, Optional, Tuple

import torch

from .. import transforms
from .vision import VisionDataset


class FakeData(VisionDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_offset: int = 0,
    ) -> None:
        super().__init__(transform=transform, target_transform=target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.item()

    def __len__(self) -> int:
        return self.size


class FakeImageFolder(FakeData):
    """
    A subclass of `FakeData` that simulates a dataset of randomly generated images, intended
    to mimic the structure and functionality of a real image folder dataset. This class is 
    useful for testing or benchmarking where access to actual image data is not necessary or 
    desirable. Images are generated on-the-fly and can be returned as PIL images. 

    Args:
        root (str): Root directory of the dataset. This parameter is kept for interface
            compatibility with other datasets but is not used since the data is fake.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g., `transforms.RandomCrop`. This
            is useful for applying data augmentation. (default: None)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. This can be used for transforming the numerical
            labels. (default: None)
        loader (callable, optional): A function to load an image given its path. This
            parameter is kept for interface compatibility with other datasets but is
            not used since the data is fake. (default: None)
        is_valid_file (callable, optional): A function that takes path of an Image file
            and checks if the file is a valid file (used to check corrupt files).
            This parameter is kept for interface compatibility with other datasets but is
            not used since the data is fake. (default: None)
        size (int, optional): The size of the dataset, i.e., how many fake images to
            generate. (default: 1000)
        image_size (Tuple[int, int, int], optional): The dimensions of the generated images
            as (channels, height, width). (default: (3, 224, 224))
        num_classes (int, optional): The number of classes in the dataset. This determines
            the range of labels for the generated targets. (default: 10)

    Examples:
        >>> dataset = FakeImageFolder(root='path/to/imaginary/dataset', transform=transforms.ToTensor())
        >>> img, label = dataset[0]
        >>> img.size()
        torch.Size([3, 224, 224])
        >>> label
        5  # Randomly generated label
    """
    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
    ):
        super().__init__(
            size=size,
            image_size=image_size,
            num_classes=num_classes,
            transform=transform,
            target_transform=target_transform,
        )