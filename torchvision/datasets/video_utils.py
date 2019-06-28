import bisect
import torch
from torchvision.io import read_video_timestamps, read_video


def unfold(tensor, size, step, dilation):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors
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

    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.

    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1):
        self.video_paths = video_paths
        self._compute_frame_pts()
        self.compute_clips(clip_length_in_frames, frames_between_clips)

    def _compute_frame_pts(self):
        self.video_pts = []
        # TODO maybe paralellize this
        for video_file in self.video_paths:
            clips = read_video_timestamps(video_file)
            self.video_pts.append(torch.as_tensor(clips))

    def compute_clips(self, num_frames, step, dilation=1):
        """
        Compute all consecutive sequences of clips from video_pts.

        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            dilation (int): distance between two consecutive frames
                in a clip
        """
        self.num_frames = num_frames
        self.step = step
        self.dilation = dilation
        self.clips = []
        for video_pts in self.video_pts:
            clips = unfold(video_pts, num_frames, step, dilation)
            self.clips.append(clips)
        l = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = l.cumsum(0).tolist()

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
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]
        video, audio, info = read_video(video_path, clip_pts[0].item(), clip_pts[-1].item())
        video = video[::self.dilation]
        # TODO change video_fps in info?
        assert len(video) == self.num_frames
        return video, audio, info, video_idx
