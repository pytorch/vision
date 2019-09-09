import bisect
from fractions import Fraction
import math
import os
import pickle
import torch
from torchvision.io import (
    read_video_timestamps_from_file,
    read_video_from_file,
)

from .utils import tqdm


def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


def pts_convert(pts, timebase_from, timebase_to, round_func=math.floor):
    """convert pts between different time bases
    Args:
        pts: presentation timestamp, float
        timebase_from: original timebase. Fraction
        timebase_to: new timebase. Fraction
        round_func: rounding function.
    """
    new_pts = Fraction(pts, 1) * timebase_from / timebase_to
    return round_func(new_pts)


class VideoClips(object):
    """
    Given a list of video files, computes all consecutive subvideos of size
    `clip_length_in_frames`, where the distance between each subvideo in the
    same video is defined by `frames_between_clips`.
    If `frame_rate` is specified, it will also resample all the videos to have
    the same frame rate, and the clips will refer to this frame rate.

    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.

    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.

    Arguments:
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        frames_between_clips (int): step (in frames) between each clip
        frame_rate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the clips will be defined
            on the resampled video
        _precomputed_metadata (dict, optional): a dictionary of dataset metadata
        _precomputed_metadata_filepath (str, optional): the path of metadata pickled file.
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1,
                 frame_rate=None, _precomputed_metadata=None, _precomputed_metadata_filepath=None):
        self.video_paths = video_paths
        # at most one of _precomputed_metadata and _precomputed_metadata_filepath can be specified
        assert _precomputed_metadata is None or _precomputed_metadata_filepath is None, \
            "_precomputed_metadata and _precomputed_metadata_filepath can not be both specified"

        if _precomputed_metadata is None and _precomputed_metadata_filepath is None:
            self._compute_frame_pts()
        elif _precomputed_metadata is not None:
            self._init_from_metadata(_precomputed_metadata)
        elif _precomputed_metadata_filepath is not None:
            self.load_metadata(_precomputed_metadata_filepath)

        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _compute_frame_pts(self):
        self.video_pts = []
        self.info = []

        # strategy: use a DataLoader to parallelize read_video_timestamps_from_file
        # so need to create a dummy dataset first
        class DS(object):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return read_video_timestamps_from_file(self.x[idx])

        import torch.utils.data
        dl = torch.utils.data.DataLoader(
            DS(self.video_paths),
            batch_size=16,
            num_workers=torch.get_num_threads(),
            collate_fn=lambda x: x)

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                video_pts, _audio_pts, info = list(zip(*batch))
                video_pts = [torch.as_tensor(c) for c in video_pts]
                self.video_pts.extend(video_pts)
                self.info.extend(info)

    def _init_from_metadata(self, metadata):
        assert len(self.video_paths) == len(metadata["video_pts"])
        assert len(self.video_paths) == len(metadata["info"])
        self.video_pts = metadata["video_pts"]
        self.info = metadata["info"]

    def load_metadata(self, filepath):
        assert os.path.exists(filepath), "File not found: %s" % filepath
        with open(filepath, 'rb') as fp:
            metadata = pickle.load(fp)
            self._init_from_metadata(metadata)

    def save_metadata(self, filepath):
        metadata = {
            "video_pts": self.video_pts,
            "info": self.info,
        }
        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            try:
                os.mkdirs(filedir)
            except Exception:
                print("Warning: fail to save metadata in folder: %s" % filedir)
                return

        with open(filepath, "wb") as fp:
            pickle.dump(metadata, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("Use pickle to save metadata to file: %s" % filepath)

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        info = [self.info[i] for i in indices]
        metadata = {
            "video_pts": video_pts,
            "info": info,
        }
        return type(self)(video_paths, self.num_frames, self.step, self.frame_rate,
                          _precomputed_metadata=metadata)

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_frames, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            dilation (int): distance between two consecutive frames
                in a clip
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, info in zip(self.video_pts, self.info):
            clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, info["video_fps"], frame_rate)
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]
        info = self.info[video_idx]

        video_start_pts = clip_pts[0].item()
        video_end_pts = clip_pts[-1].item()

        audio_start_pts, audio_end_pts = 0, -1
        audio_timebase = Fraction(0, 1)
        if "audio_timebase" in info:
            audio_timebase = info["audio_timebase"]
            audio_start_pts = pts_convert(
                video_start_pts,
                info["video_timebase"],
                info["audio_timebase"],
                math.floor,
            )
            audio_end_pts = pts_convert(
                video_start_pts,
                info["video_timebase"],
                info["audio_timebase"],
                math.ceil,
            )
        video, audio, info = read_video_from_file(
            video_path,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase=info["video_timebase"],
            audio_pts_range=(audio_start_pts, audio_end_pts),
            audio_timebase=audio_timebase,
        )
        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, "{} x {}".format(video.shape, self.num_frames)
        return video, audio, info, video_idx
