import urllib
import time
import os
import warnings


from os import path
import csv
from typing import Callable, Optional
from functools import partial
from multiprocessing import Pool

from .utils import download_and_extract_archive, download_url
from .folder import find_classes, make_dataset
from .video_utils import VideoClips
from .vision import VisionDataset


def _dl_wrap(tarpath, videopath, line):
    download_and_extract_archive(line, tarpath, videopath)


class Kinetics(VisionDataset):
    """` Generic Kinetics <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400/600/700 are action recognition video datasets.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Args:
        root (string): Root directory of the (split of the) Kinetics Dataset. 
            Directory should be structured as follows:
            .. code::

                root/
                ├── class1
                │   ├── clip1.avi
                │   ├── clip2.avi
                │   └── ...
                └── class2
                    ├── clipx.avi
                    └── ...
            If the split is not defined, it is appended using the split argument.
        n_classes (int): select between Kinetics-400, Kinetics-600, and Kinetics-700 
        split (str): split of the dataset to consider; currently supports ["train", "val"]
        frame_rate (float): If not None, interpolate different frame rate for each clip.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        annotation_path (str): path to official Kinetics annotation file.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        download (bool): Download the official version of the dataset to root folder.
        num_workers (int): Use multiple workers for VideoClips creation
        _num_download_workers (int): Use multiprocessing in order to speed up download.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): the `T` video frames in torch.uint8 tensor
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points in torch.float tensor
            - label (int): class of the video clip
    
    Raises:
        RuntimeError: If ``download is True`` and the image archive is already extracted.
    """

    _FILES = {
        "400": "https://s3.amazonaws.com/kinetics/400/{split}/k400_{split}_path.txt",
        "600": "https://s3.amazonaws.com/kinetics/600/{split}/k600_{split}_path.txt",
        "700": "https://s3.amazonaws.com/kinetics/700_2020/{split}/k700_2020_{split}_path.txt",
    }
    _ANNOTATION = {
        "400": "https://s3.amazonaws.com/kinetics/400/annotations/{split}.csv",
        "600": "https://s3.amazonaws.com/kinetics/600/annotations/{split}.txt",
        "700": "https://s3.amazonaws.com/kinetics/700_2020/annotations/{split}.csv",
    }

    def __init__(
        self,
        root: str,
        num_classes: str = "400",
        split: str = "train",
        frame_rate: float = None,
        frames_per_clip: int = 5,
        step_between_clips: int = 1,
        annotation_path: str = None,
        transform: Optional[Callable] = None,
        extensions=("avi", "mp4"),
        download: bool = False,
        num_workers: int = 1,
        _precomputed_metadata=None,
        _num_download_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ) -> None:

        # TODO: support test
        assert split in ["train", "val"]
        assert num_classes in ["400", "600", "700"]
        self.n_classes = num_classes
        self.extensions = extensions
        self._num_download_workers = _num_download_workers

        self._set_up_paths(root, split)

        if annotation_path is not None:
            self.annotations = annotation_path

        if download:
            self.download_and_process_videos()
        # init folder dataset at the end
        super().__init__(self.root)

        # and then figure out the rest
        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None
        )
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )
        self.transform = transform

    def _set_up_paths(self, root, split) -> None:
        """Sets up self.root and self.split to avoid confusion.
        Split in the root (e.g. kinetics/val) overrides the setting in 
        the split.
        """
        self.split = split
        if path.basename(root) == split:
            self.root = root
        elif path.basename(root) in ["train", "val"]:
            self.root = root
            self.split = path.basename(root)
            warnings.warn(
                f"Root {root} points to a different split than {split}."
                f"Assigning self.split to {self.split}."
            )
        else:
            self.root = path.join(root, split)

    def download_and_process_videos(self) -> None:
        """
        downloads all the videos to the _root_ folder
        in the expected format
        """
        tic = time.time()
        self._download_videos()
        toc = time.time()
        print("Elapsed time for downloading in mins ", (toc - tic) / 60)
        self._make_ds_structure()
        toc2 = time.time()
        print("Elapsed time for processing in mins ", (toc2 - toc) / 60)
        print("Elapsed time overall in mins ", (toc2 - tic) / 60)

    def _download_videos(self) -> None:
        """download tarballs containing the video to 
        "tars" folder and extract them into the _split_ folder
        where split is one of the official dataset splits.

        Raises:
            RuntimeError: if download folder exists, break to prevent
              downloading entire dataset again.
        """
        if path.exists(self.root):
            raise RuntimeError(
                f"The directory {self.root} already exists. If you want to re-download or re-extract the images, "
                f"delete the directory."
            )

        file_url = urllib.request.urlopen(
            self._FILES[self.n_classes].format(split=self.split)
        )
        kinetics_dir, _ = path.split(self.root)
        tar_path = path.join(kinetics_dir, "tars")
        annotation_path = path.join(kinetics_dir, "annotations")

        # download annotations
        download_url(
            self._ANNOTATION[self.n_classes].format(split=self.split), annotation_path
        )
        self.annotations = os.path.join(annotation_path, f"{self.split}.csv")

        if self._num_download_workers == 1:
            for line in file_url:
                line = str(line.decode("utf-8")).replace("\n", "")
                download_and_extract_archive(line, tar_path, self.root)
        else:
            part = partial(_dl_wrap, tar_path, self.root)
            lines = [str(line.decode("utf-8")).replace("\n", "") for line in file_url]
            poolproc = Pool(self._num_download_workers)
            poolproc.map(part, lines)

    def _make_ds_structure(self):
        """move videos from 
        root/
            ├── clip1.avi
            ├── clip2.avi
        
        to the correct format as described below:
        root/
            ├── class1
            │   ├── clip1.avi

        """
        file_tmp = "{ytid}_{start:06}_{end:06}.mp4"
        with open(self.annotations) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                f = file_tmp.format(
                    ytid=row["youtube_id"],
                    start=int(row["time_start"]),
                    end=int(row["time_end"]),
                )
                label = (
                    row["label"]
                    .replace(" ", "_")
                    .replace("'", "")
                    .replace("(", "")
                    .replace(")", "")
                )
                os.makedirs(os.path.join(self.root, label), exist_ok=True)
                existing_file = os.path.join(self.root, f)
                if os.path.isfile(existing_file):
                    os.replace(
                        existing_file, os.path.join(self.root, label, f),
                    )

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label


class Kinetics400(Kinetics):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset. Should be structured as follows:

            .. code::

                root/
                ├── class1
                │   ├── clip1.avi
                │   ├── clip2.avi
                │   └── ...
                └── class2
                    ├── clipx.avi
                    └── ...

        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): the `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip
    """

    def __init__(
        self,
        root,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        extensions=("avi",),
        transform=None,
        _precomputed_metadata=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ):
        warnings.warn(
            "torchvision now supports multiple versions of Kinetics"
            "datasets, available via Kinetics class with a separate "
            "n_classes parameter. This function might get deprecated in the future."
        )

        super(Kinetics400, self).__init__(
            root=root,
            num_classes="400",
            frame_rate=frame_rate,
            step_between_clips=step_between_clips,
            frames_per_clip=frames_per_clip,
            extensions=extensions,
            transform=transform,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_channels=_audio_channels,
            _audio_samples=_audio_samples,
            download=False,
        )
