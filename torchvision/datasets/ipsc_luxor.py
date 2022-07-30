import os
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple, Union, List, Dict

class IPSCLuxor(VisionDataset):
    """`IPSC Luxor Catch-ya! <https://ipsc.ksp.sk/2016/real/problems/l.html>`_ Dataset.
    Args:
        root (string, optional): Root directory of dataset where directory
            ``ipsc_luxor`` exists or will be saved to if download is set to True.
        subproblem (string or list): Which subproblems to use: ``alphabet``,
            ``l1``, or ``l2``. Defaults to ``l2``.
        train (bool, optional):
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "ipsc_luxor"
    base_url = "https://ipsc.ksp.sk/2016/real/"

    resources = [
        (base_url + "problems/l.zip", "md5"),
        (base_url + "solutions/l1.out", "md5"),
        (base_url + "solutions/l2.out", "md5"),
    ]


    def __init__(
        self,
        root: str,
        subproblem: Union[List[str], str] = "l2",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "ipsc_luxor"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(subproblem, str):
            subproblem = [subproblem]
        self.subproblem = [verify_str_arg(p, "subproblem", ("alphabet", "l1", "l2")) for p in subproblem]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.classes = list("abcdegijklmnopqrstuvwyz") # no "f", "h", or "x"


    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        for filename, md5 in self.resources[0:]:
            fpath = os.path.join(self.root, self.base_folder, filename)
            print("check", fpath)
            if not check_integrity(fpath):
                return False
        return True


    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(IPSCLuxor.resources[0][0], self.root)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}
