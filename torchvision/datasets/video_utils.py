import bisect
import torch
import torch.utils.data
from torchvision.io import read_video_timestamps, read_video


def unfold(tensor, size, step, dilation):
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
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1,
                 frame_rate=None):
        self.video_paths = video_paths
        self._compute_frame_pts()
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []
        # TODO maybe paralellize this
        for video_file in self.video_paths:
            clips, fps = read_video_timestamps(video_file)
            self.video_pts.append(torch.as_tensor(clips))
            self.video_fps.append(fps)

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
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            if frame_rate is not None:
                # divup, == int(ceil(fps / frame_rate))
                dilation = max(int((fps + frame_rate - 1) // frame_rate), 1)
            else:
                dilation = 1
            clips = unfold(video_pts, num_frames, step, dilation)
            self.clips.append(clips)
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

    def resample_video(self, video, original_fps, new_fps):
        """
        Resamples a video so that it matches a new fps specified
        """
        step = original_fps / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return video[::step]
        idxs = torch.arange(self.num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        video = video[idxs]
        return video

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
        video, audio, info = read_video(video_path, clip_pts[0].item(), clip_pts[-1].item())
        if self.frame_rate is not None:
            fps = self.video_fps[video_idx]
            video = self.resample_video(video, fps, self.frame_rate)
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames
        return video, audio, info, video_idx


class RandomClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """
    def __init__(self, video_clips, max_clips_per_video):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs))
        idxs = idxs[perm].tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)
