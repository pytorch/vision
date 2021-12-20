import os
import os.path
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

from PIL import Image

from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset


class OxfordIIITPet(VisionDataset):
    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_type: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_type, str):
            target_type = [target_type]
        self._target_type = [verify_str_arg(t, "target_type", self._TARGET_TYPES) for t in target_type]

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(os.path.join(self._anns_folder, f"{self._split}.txt")) as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
        self._segmentations = [os.path.join(self._segmentations_folder, f"{image_id}.png") for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target = []
        for t in self._target_type:
            if t == "category":
                target.append(self._labels[idx])
            else:  # t == "segmentation"
                target.append(Image.open(self._segmentations[idx]))

        target: Any
        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def _base_folder(self):
        return os.path.join(self.root, "oxford-iiit-pet")

    @property
    def _images_folder(self) -> str:
        return os.path.join(self._base_folder, "images")

    @property
    def _anns_folder(self) -> str:
        return os.path.join(self._base_folder, "annotations")

    @property
    def _segmentations_folder(self) -> str:
        return os.path.join(self._anns_folder, "trimaps")

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)
