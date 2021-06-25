from PIL import Image
import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from .vision import VisionDataset
from .utils import download_and_extract_archive, verify_str_arg

CATEGORIES = ["kingdom", "phylum", "class", "order", "family", "genus"]


class INaturalist(VisionDataset):
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.

    Args:
        root (string): Root directory of dataset where the image files are stored.
            This class does not require/use annotation files.
        target_type (string or list, optional): Type of target to use, one of:
            * ``full``: the full category (species)
            * ``kingdom``: e.g. "Animalia"
            * ``phylum``: e.g. "Arthropoda"
            * ``class``: e.g. "Insecta"
            * ``order``: e.g. "Coleoptera"
            * ``family``: e.g. "Cleridae"
            * ``genus``: e.g. "Trichodes"
            Can also be a list to output a tuple with all specified target types.
            Defaults to ``full``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
            self,
            root: str,
            target_type: Union[List[str], str] = "full",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(INaturalist, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("full", *CATEGORIES))
                            for t in target_type]

        if not os.path.exists(self.root):
            raise RuntimeError(f'Dataset not found in {self.root}. You need to download it.')
        self.all_categories = sorted(os.listdir(self.root))
        if (not self.all_categories):
            raise RuntimeError(f'Dataset not found in {self.root}. You need to download it.')

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {
            k: {} for k in CATEGORIES
        }
        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split('_')
            if len(pieces) != 8:
                raise RuntimeError(f'Unexpected category name {dir_name}, wrong number of pieces')
            if pieces[0] != f'{dir_index:05d}':
                raise RuntimeError(f'Unexpected category id {pieces[0]}, expecting {dir_index:05d}')
            cat_map = {}
            for cat, name in zip(CATEGORIES, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)

            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root,
                                      self.all_categories[cat_id],
                                      fname))

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise RuntimeError(f"Invalid category type {category_type}")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise RuntimeError(f"Invalid category id {category_id} for {category_type}")
