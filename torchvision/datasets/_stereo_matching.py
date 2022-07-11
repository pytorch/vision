from abc import ABC, abstractmethod
from functools import reduce
from glob import glob
from pathlib import Path
from random import random
import re
import shutil
from typing import Callable, List, Optional, Tuple, Any
import lzma
from torch import Tensor
from .vision import VisionDataset
from .utils import download_and_extract_archive, download_url, verify_str_arg
import os
from torch.utils.model_zoo import tqdm
import numpy as np
from PIL import Image

__all__ = (
    "CSEStereo"
    "Middlebury2014"
    "ETH3D"
    "Kitti2012"
    "Kitti2015"
)


def read_pfm_file(file_path: str) -> np.array:
    # adapted from https://github.com/ucbdrive/hd3/blob/master/utils/pfm.py
    with open(file_path, "rb") as file:
        header = file.readline().rstrip()
        assert header in [b"PF", b"Pf"], f"{file_path} is not a valid .pfm file"

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        assert dim_match, f"{file_path} has a Malformed PFM header"

        width, height = map(int, dim_match.groups())
        channels = 3 if header == "PF" else 1
        scale = float(file.readline().rstrip())
        # check for endian type
        if scale < 0:
            scale = -scale
            endian = '<'
        else:
            endian = '>'

        data = np.fromfile(file, endian + 'f')
        data = np.reshape(data, (height, width, channels))
        data = np.flipud(data)

        return data


class StereoMatchingDataset(ABC, VisionDataset):
    """Base interface for Stereo matching datasets"""

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root=root)
        self.transforms = transforms

        self._images: List[Tuple] = []
        self._disparities: List[Tuple] = []

    def _read_img(self, file_path: str) -> Image.Image:
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @abstractmethod
    def _read_disparity(self, file_path: str) -> Tuple:
        # function that returns a disparity map and an occlusion map
        pass

    def __getitem__(self, index: int) -> Tuple[Tuple, Tuple, Tuple]:
        img_left = self._read_img(self._images[index][0])
        img_right = self._read_img(self._images[index][1])

        dsp_map_left, occ_mask_left = self._read_disparity(self._disparities[index][0])
        dsp_map_right, occ_mask_right = self._read_disparity(self._disparities[index][1])

        imgs = (img_left, img_right)
        dsp_maps = (dsp_map_left, dsp_map_right)
        occ_masks = (occ_mask_left, occ_mask_right)

        if self.transforms is not None:
            imgs, dsp_maps, occ_masks, = self.transforms(imgs, dsp_maps, occ_masks)

        return imgs, dsp_maps, occ_masks

    def __len__(self) -> int:
        return len(self._images)


class CRESSyntethicStereo(StereoMatchingDataset):
    """Synthetic dataset used in training the `CREStereo <https://arxiv.org/pdf/2203.11483.pdf>`_ architecture. 

   Ported from the download script in the paper github `repo <https://github.com/megvii-research/CREStereo>`_.
   """
    DOWNLOAD_SPACE = 4 * 1024 * 1024 * 1024  # dataset requires download requires about 400 GB of free space

    EXPERIMENTAL_RANGE = 1  # TODO: remove after validating dataset structure / flow

    def __init__(self, root: str, transforms: Optional[Callable] = None, download: bool = True):
        super().__init__(root, transforms)
        # if the API user requests a dataset download check that the user can download it
        if download:
            statvfs = os.statvfs(root)
            # measured in bytes
            available_space = statvfs.f_frsize * statvfs.f_bavail
            if available_space - self.DOWNLOAD_SPACE < 0:
                raise ValueError(
                    f"The storage device for {root} is too small to download the dataset), "
                    f"an additional {self.DOWNLOAD_SPACE - self.available_space:.2f} GB are required."
                )
            self._download_dataset(root)

    def _download_dataset(self, root: str) -> None:
        # TODO: remove before release, used only for testing purposes
        dirs = ["tree", "shapenet", "reflective", "hole"]
        # create directory subtree for the download
        for d in dirs:
            d_path = os.path.join(root, d)
            if not os.path.exists(d_path):
                os.makedirs(d_path)

            for i in range(self.EXPERIMENTAL_RANGE):
                url = f"https://data.megengine.org.cn/research/crestereo/dataset/{d}/{i}.tar"
                download_and_extract_archive(url=url, download_root=d_path, remove_finished=True)


class Middlebury2014(StereoMatchingDataset):
    """Publicly available scenes from the Middlebury dataset `2014 version <https://vision.middlebury.edu/stereo/data/scenes2014/>`.

    The dataset mostly follows the original format, without containing the ambient subdirectories.  : ::

        root
            Middlebury2014
                train
                    scene1-{ ,perfect,imperfect}
                        calib.txt                    
                        im{0,1}.png                  
                        im1E.png                     
                        im1L.png                     
                        disp{0,1}.pfm                
                        disp{0,1}-n.png              
                        disp{0,1}-sd.pfm             
                        disp{0,1}y.pfm
                    scene2-{ ,perfect,imperfect}
                        calib.txt                    
                        im{0,1}.png                  
                        im1E.png                     
                        im1L.png                     
                        disp{0,1}.pfm                
                        disp{0,1}-n.png              
                        disp{0,1}-sd.pfm             
                        disp{0,1}y.pfm
                    ...
                additional
                    scene1-{ ,perfect,imperfect}
                        calib.txt                    
                        im{0,1}.png                  
                        im1E.png                     
                        im1L.png                     
                        disp{0,1}.pfm                
                        disp{0,1}-n.png              
                        disp{0,1}-sd.pfm             
                        disp{0,1}y.pfm
                    ...
                test
                    scene1
                        calib.txt
                        im{0,1}.png
                    scene2
                        calib.txt
                        im{0,1}.png
                    ...


    Args:
        root (string): Root directory of the Middleburry 2014 Dataset.
        split (string, optional): The dataset split of scenes, either "train" (default), test, or "additional"
        use_ambient_views (boolean, optional): Whether to use different expose or lightning views when possible. Sampled with equal probability.
        calibration (string, optional): Wether or not to use the calibrated (default) or uncalibrated scenes. 
        transforms (callalbe, optional): A function/transform that takes in
            ``left_img, right_img, left_disparity, right_disparity`` and returns a transformed version.
        download (boolean, optional): Wether or not to download the dataset in the ``root`` directory. 
    """

    splits = {
        "train": ["Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable", "Recycle", "Shelves", "Vintage"],
        "additional": ["Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Mask", "Shopvac", "Sticks", "Storage", "Sword1", "Sword2", "Umbrella"],
        "test": ['Plants', 'Classroom2E', 'Classroom2', 'Australia', 'DjembeL', 'CrusadeP', 'Crusade', 'Hoops', 'Bicycle2', 'Staircase', 'Newkuba', 'AustraliaP', 'Djembe', 'Livingroom', 'Computer']
    }

    def __init__(
        self,
        *,
        root: str,
        split: str = "train",
        use_ambient_views: bool = False,
        transforms: Optional[Callable] = None,
        download: bool = False
    ):
        super().__init__(root, transforms)
        verify_str_arg(split, "split", valid_values=("train", "test", "additional"))

        if download:
            self._download_dataset(root)

        root = Path(root) / "FlyingChairs"
        if not os.path.exists(root / split):
            raise FileNotFoundError(
                f"The {split} directory was not found in the provided root directory"
            )

        split_scenes = self.splits[split]
        # check that the provided root folder contains the scene splits
        if not all(s in os.listdir(root / split) for s in split_scenes):
            raise FileNotFoundError(f"Provided root folder does not contain any scenes from the {split} split.")

        imgs_left = sorted(glob(str(root / split / "*" / "im0.png")))
        imgs_right = sorted(glob(str(root / split / "*" / "im1.png")))
        self._images = list((l, r) for l, r in zip(imgs_left, imgs_right))

        if split == "test":
            dsp_maps_left, dsp_maps_right = list("" for _ in imgs_left), list("" for _ in imgs_right)
        else:

            dsp_maps_left = sorted(glob(str(root / split / "*" / "disp0.pfm")))
            dsp_maps_right = sorted(glob(str(root / split / "*" / "disp1.pfm")))
        self._disparities = list((l, r) for l, r in zip(dsp_maps_left, dsp_maps_right))

        self.use_ambient_views = use_ambient_views

    def __getitem__(self, index: int) -> Tuple:
        return super().__getitem__(index)

    def _read_img(self, file_path: str) -> Image.Image:
        if os.path.basename(file_path) == "im1.png" and self.use_ambient_views:
            # initialize sampleable container
            ambient_file_paths = list(file_path.replace("im1.png", view_name) for view_name in ["im1E.png", "im1L.png"])
            # double check that we're not going to try to read from an invalid file path
            ambient_file_paths = list(filter(lambda p: os.path.exists(p), ambient_file_paths))
            # keep the original image as an option as well for uniform sampling between base views
            ambient_file_paths.append(file_path)
            file_path = random.choice(ambient_file_paths)
        return super()._read_img(file_path)

    def _read_disparity(self, file_path: str) -> Tuple:
        if not os.path.exists(file_path):  # case when dealing with the test split
            return None, None
        dsp_mask = read_pfm_file(file_path)
        occ_mask = dsp_mask < 1e3
        return dsp_mask, occ_mask

    def _download_dataset(self, root: str):
        base_url = "https://vision.middlebury.edu/stereo/data/scenes2014/zip"
        # train and additional splits have 2 different calibration settings
        root = Path(root) / "Middlebury2014"
        for split_name, split_scenes in self.splits.values():
            if split_name == "test":
                continue
            split_root = root / split_name
            for scene in split_scenes:
                scene_name = f"{scene}-{calibration}"
                for calibration in ["perfect", "imperfect"]:
                    scene_url = f"{base_url}/{scene_name}.zip"
                    download_and_extract_archive(url=scene_url, filename=scene_name, download_root=str(split_root), remove_finished=True)

        if any(s not in os.listdir(root) for s in self.splits["test"]):
            # test split is downloaded from a different location
            test_set_url = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip"

            # the unzip is going to produce a directory MiddEval3 with two subdirectories trainingF and testF
            # we want to move the contents from testF into the  directory
            download_and_extract_archive(url=test_set_url, download_root=root, remove_finished=True)
            for scene_dir, scene_names, _ in os.walk(str(root / "MiddEval3/testF")):
                for scene in scene_names:
                    shutil.move(os.path.join(scene_dir, scene), os.path.join(root, scene))

            # cleanup MiddEval3 directory
            shutil.rmtree(os.path.join(root, "MiddEval3"))


class ETH3D(StereoMatchingDataset):
    """"ETH3D `Low-Res Two-View <https://www.eth3d.net/datasets>`_ dataset.

    The dataset is expected to have the following structure: ::

        root
            ETH3D
                two_view_training
                    scene1
                        im1.png 
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    scene2
                        im1.png 
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    ...
                two_view_training_gt
                    scene1
                        disp0GT.pfm
                        mask0nocc.png
                    scene2
                        disp0GT.pfm
                        mask0nocc.png
                    ...
                two_view_testing
                    scene1
                        im1.png 
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    scene2
                        im1.png 
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    ...

    Args:
        root (string): Root directory of the ETH3D Dataset.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        calibration (string, optional): Wether or not to use the calibrated (default) or uncalibrated scenes. 
        transforms (callalbe, optional): A function/transform that takes in
            ``left_img, right_img, left_disparity, right_disparity`` and returns a transformed version.
    """

    def __init__(self, root: str, split: str = "train", transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))

        root = Path(root) / "ETH3D"
        img_dir = "two_view_training" if split == "train" else "two_view_testing"
        anot_dir = "two_view_training_gt"

        imgs_left = sorted(glob(str(root / img_dir / "*" / "*im0.png")))
        imgs_right = sorted(glob(str(root / img_dir / "*" / "*im1.png")))

        if split == "test":
            dsp_masks_left, dsp_masks_right = list("" for _ in imgs_left), list("" for _ in imgs_right)
        else:
            dsp_masks_left = sorted(glob(str(root / anot_dir / "*" / "*[0-1].pfm")))
            # no masks for the right view, always using left as reference
            dsp_masks_right = list("" for _ in dsp_masks_left)

        self._images = list((l, r) for l, r in zip(imgs_left, imgs_right))
        self._dsp_masks = list((l, r) for l, r in zip(dsp_masks_left, dsp_masks_right))

    def _read_disparity(self, file_path: str) -> Tuple:
        if not os.path.exists(file_path):
            return None, None

        dsp_mask = read_pfm_file(file_path)
        occ_mask = Image.open(file_path.replace("disp0GT.pfm", "mask0nocc.png"))
        occ_mask = np.array(occ_mask)
        return dsp_mask, occ_mask

    def __getitem__(self, index: int) -> Tuple[Tuple, Tuple, Tuple]:
        return super().__getitem__(index)


class Kitti2012(StereoMatchingDataset):
    """"Kitti dataset from the `2012 <http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php>`_ stereo evaluation benchmark.
    Uses the RGB images for consistency with Kitti 2015.

    The dataset is expected to have the following structure: ::

        root
            Kitti2012
                testing
                    colored_0
                    colored_1
                training
                    colored_0
                    colored_1
                    disp_noc
                    calib

    Args:
        root (string): Root directory where Kitti2012 is located.
        split (string, optional): The dataset split of scenes, either "train" (default), test, or "additional"
        transforms (callalbe, optional): A function/transform that takes in
            ``left_img, right_img, left_disparity, right_disparity`` and returns a transformed version.
        download (boolean, optional): Wether or not to download the dataset in the ``root`` directory. 
    """

    def __init__(self, root: str, split: str = "train", transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))

        root = Path(root) / "Kitti2012" / (split + "ing")
        imgs_left = sorted(glob(str(root / "colored_0" / "*_10.png")))
        imgs_right = sorted(glob(str(root / "colored_1" / "*_10.png")))

        if split == "train":
            dsp_masks_left = sorted(glob(str(root / "disp_noc" / "*.png")))
            dsp_masks_right = list("" for _ in dsp_masks_left)
        else:
            dsp_masks_left, dsp_masks_right = list("" for _ in dsp_masks_left), list("" for _ in dsp_masks_right)

        self._images = list((l, r) for l, r in zip(imgs_left, imgs_right))
        self._dsp_masks = list((l, r) for l, r in zip(dsp_masks_left, dsp_masks_right))

    def _read_disparity(self, file_path: str) -> Tuple:
        if not os.path.exists(file_path):
            return None, None

        dsp_mask = np.array(Image.open(file_path)) / 256.0
        occ_mask = dsp_mask > 0.0

        return dsp_mask, occ_mask

    def __getitem__(self, index: int) -> Tuple[Tuple, Tuple, Tuple]:
        return super().__getitem__(index)


class Kitti2015(StereoMatchingDataset):
    """"Kitti dataset from the `2015 <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php>`_ stereo evaluation benchmark.

    The dataset is expected to have the following structure: ::

        root
            Kitti2015
                testing
                    image_2
                    image_3
                training
                    image_2
                    image_3
                    disp_noc_0
                    disp_noc_1
                    calib

    Args:
        root (string): Root directory where Kitti2015 is located.
        split (string, optional): The dataset split of scenes, either "train" (default) or test.
        transforms (callalbe, optional): A function/transform that takes in
            ``left_img, right_img, left_disparity, right_disparity`` and returns a transformed version.
    """

    def __init__(self, root: str, split: str = "train", transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        verify_str_arg(split, "split", valid_values=("train", "test"))

        root = Path(root) / "Kitti2015" / (split + "ing")
        imgs_left = sorted(glob(str(root / "image_2" / "*_10.png")))
        imgs_right = sorted(glob(str(root / "image_3" / "*_10.png")))

        if split == "train":
            dsp_masks_left = sorted(glob(str(root / "disp_noc_0" / "*.png")))
            dsp_masks_right = sorted(glob(str(root / "disp_noc_1" / "*.png")))
        else:
            dsp_masks_left, dsp_masks_right = list("" for _ in dsp_masks_left), list("" for _ in dsp_masks_right)

        self._images = list((l, r) for l, r in zip(imgs_left, imgs_right))
        self._dsp_masks = list((l, r) for l, r in zip(dsp_masks_left, dsp_masks_right))

    def _read_disparity(self, file_path: str) -> Tuple:
        if not os.path.exists(file_path):
            return None, None

        dsp_mask = np.array(Image.open(file_path)) / 256.0
        occ_mask = dsp_mask > 0.0

        return dsp_mask, occ_mask

    def __getitem__(self, index: int) -> Tuple[Tuple, Tuple, Tuple]:
        return super().__getitem__(index)


class SintelDataset(StereoMatchingDataset):
    """"Sintel `Stereo Dataset <http://sintel.is.tue.mpg.de/stereo>`_.

    Args:
        root (string): Root directory where Sintel Stereo is located.
        transforms (callalbe, optional): A function/transform that takes in
            ``left_img, right_img, left_disparity, right_disparity`` and returns a transformed version.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        root = Path(root) / "Sintel"

        imgs_left = sorted(glob(str(root / "training" / "final_left" / "*" / "*.png")))
        imgs_right = sorted(glob(str(root / "training" / "final_right" / "*" / "*.png")))

        dps_masks_left = sorted(glob(str(root / "training" / "disparities" / "*" / "*.png")))
        dsp_masks_right = list("" for _ in dps_masks_left)

        self._images = list((l, r) for l, r in zip(imgs_left, imgs_right))
        self._dsp_masks = list((l, r) for l, r in zip(dps_masks_left, dsp_masks_right))

    def _read_disparity(self, file_path: str) -> Tuple:
        if not os.path.exists(file_path):
            return None, None

        # disparity decoding as per Sintel instructions
        dsp_mask = np.array(Image.open(file_path), dtype=np.float32)
        r, g, b = np.split(dsp_mask, 3, axis=-1)
        dsp_mask = r * 4 + g / (2**6) + b / (2**14)

        # occlusion mask
        occ_mask = np.array(Image.open(file_path.replace("disparities", "occlusions"))) > 0
        # out of frame mask
        off_mask = np.array(Image.open(file_path.replace("disparities", "outofframe"))) > 0
        # combine the masks together
        occ_mask = np.logical_or(off_mask, occ_mask)
        return dsp_mask, occ_mask

    def __getitem__(self, index: int) -> Tuple[Tuple, Tuple, Tuple]:
        return super().__getitem__(index)
