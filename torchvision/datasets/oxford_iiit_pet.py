import os
import os.path
import pathlib
import torch
from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from PIL import Image

from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
from .voc import VOCDetection


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``binary-category`` (int): Binary label for cat or dog.
                - ``detection`` (dict): Pascal VOC annotation dict with classes covering the 37 pet breeds
                - ``binary-detection`` Pascal VOC annotation dict with binary cat/dog classes
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "binary-category", "detection", "binary-detection", "segmentation")

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"
        self._xmls_folder = self._anns_folder / "xmls"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        self._bin_labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, bin_label, _ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)
                self._bin_labels.append(int(bin_label) - 1)

        self.bin_classes = ["Cat", "Dog"]
        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.bin_class_to_idx = dict(zip(self.bin_classes, range(len(self.bin_classes))))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]
        self._xmls = [self._xmls_folder / f"{image_id}.xml" for image_id in image_ids]

        # The oxford pet dataset has detection XMLs in VOC format, but some images do not have xmls
        # Here we filter to only samples that have corresponding xml files when detection is selected
        if "detection" in target_types or "binary-detection" in target_types:
            # Notify users this is not a complete dataset
            print('Dataset does not contain detection annotations for every sample. Filtering to include' \
                  ' only those that do.') # TODO: Is a simple print the right call here?
            # Set up filtered arrays
            self._labels = [lbl for lbl,xml_file in zip(self._labels,self._xmls) if os.path.isfile(xml_file)]
            self._bin_labels = [lbl for lbl,xml_file in zip(self._bin_labels,self._xmls) if os.path.isfile(xml_file)]
            self._images = [img for img,xml_file in zip(self._images,self._xmls) if os.path.isfile(xml_file)]
            self._segs = [seg for seg,xml_file in zip(self._segs,self._xmls) if os.path.isfile(xml_file)]
            self._xmls = [xml_file for xml_file in self._xmls if os.path.isfile(xml_file)]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            elif target_type == "binary-category":
                target.append(self._bin_labels[idx])
            elif target_type == "detection":
                target.append(self._to_rcnn(VOCDetection.parse_voc_xml(ET_parse(self._xmls[idx]).getroot() \
                                                                       ),self._labels[idx]))
                #target[-1]['annotation']['object'][0]['name'] = self.classes[self._labels[idx]]
            elif target_type == "binary-detection":
                target.append(self._to_rcnn(VOCDetection.parse_voc_xml(ET_parse(self._xmls[idx]).getroot() \
                                                                       ),self._bin_labels[idx]))
                #target[-1]['annotation']['object'][0]['name'] = self.classes[self._bin_labels[idx]]
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

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
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)

    def _to_rcnn(self, anno_dict: dict,label: int) -> dict:
        # Create output tensors
        out = {'boxes': torch.empty(1,4,dtype=torch.float32),
               'labels':torch.empty(1,dtype=torch.int64)}
        # Populate output
        out['boxes'][0,0] = float(anno_dict['annotation']['object'][0]['bndbox']['xmin'])
        out['boxes'][0,1] = float(anno_dict['annotation']['object'][0]['bndbox']['ymin'])
        out['boxes'][0,2] = float(anno_dict['annotation']['object'][0]['bndbox']['xmax'])
        out['boxes'][0,3] = float(anno_dict['annotation']['object'][0]['bndbox']['ymax'])
        out['labels'][0] = label

        return out