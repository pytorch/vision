import os

from .utils import list_dir
from .folder import make_dataset
from .video_utils import VideoClips
from .vision import VisionDataset


class UCF101(VisionDataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
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
        root (string): Root directory of the UCF101 Dataset.
        annotation_path (str): path to the folder containing the split files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 frame_rate=None, fold=1, train=True, transform=None,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(UCF101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.fold = fold
        self.train = train
        self.transform = transform

        # Create class to index mapping with sorted class names
        self.classes = list(sorted(list_dir(root)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Iterate through root directory to retrieve the path and the labels
        # for each dataset example
        self.samples = make_dataset(
            self.root, class_to_idx, ('avi',), is_valid_file=None)
        
        # Get the video paths that belong to the selected fold and split
        _video_paths_in_fold = self._fold_paths(annotation_path, fold, train)
        # Filter the dataset samples so only the video paths belonging to the 
        # selected fold are processed
        self.samples = [o for o in self.samples if o[0] in _video_paths_in_fold]

        # At this point, only the needed videos' path are selected
        self.video_clips = VideoClips(
            [x[0] for x in self.samples],
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.video_clips_metadata = self.video_clips.metadata

    @property
    def metadata(self):
        return self.video_clips_metadata

    def _fold_paths(self, annotation_path, fold, train):
        split = 'train' if train else 'test'
        name = f'{split}list{fold:02d}.txt'
        f = os.path.join(annotation_path, name)

        with open(f, "r") as fid:
            video_files = fid.readlines()
            video_files = [o.strip().split(" ")[0] for o in video_files]
            video_files = [os.path.join(self.root, o) for o in video_files]
            video_files = set(video_files)
        return video_files

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label
