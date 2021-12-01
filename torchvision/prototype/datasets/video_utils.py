from typing import Any, Dict, Iterator
import random
import av
import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchvision.io import video, _video_opt


class AVKeyframeReader(IterDataPipe[Dict[str, Any]]):
    def __init__(self, video_dp: IterDataPipe[Dict[str, Any]]) -> None:
        """TorchData Iterdatapype that takes in video datapipe
        and yields all keyframes in a video

        Args:
            video_dp (IterDataPipe[Dict[str, Any]]): Video dataset IterDataPipe
        """
        self.datapipe = video_dp

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for video_d in self.datapipe:
            buffer = video_d.pop("file")
            with av.open(buffer, metadata_errors="ignore") as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'
                for frame in container.decode(stream):
                    img = frame.to_image()
                    yield dict(
                        video_d,
                        frame=img,
                        pts=frame.pts,
                        video_meta={
                            "time_base": float(frame.time_base),
                            "guessed_fps": float(stream.guessed_rate),
                        })


class AVRandomFrameReader(IterDataPipe[Dict[str, Any]]):
    def __init__(self, video_dp: IterDataPipe[Dict[str, Any]], num_samples=1) -> None:
        """TorchData Iterdatapype that takes in video datapipe
        and yields `num_samples` random frames from a video.

        Args:
            video_dp (IterDataPipe[Dict[str, Any]]): Video dataset IterDataPipe
            num_samples (int, optional): Number of frames to sample from each video. Defaults to 1.
        """
        self.datapipe = video_dp
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for video_d in self.datapipe:
            buffer = video_d.pop("file")
            with av.open(buffer, metadata_errors="ignore") as container:
                stream = container.streams.video[0]
                # duration is given in time_base units as int
                duration = stream.duration
                # seek to a random frame
                seek_idxs = random.sample(list(range(duration)), self.num_samples)
                for i in seek_idxs:
                    container.seek(i, any_frame=True, stream=stream)
                    frame = next(container.decode(stream))
                    img = frame.to_image()

                    video_meta = {"time_base": float(frame.time_base),
                                  "guessed_fps": float(stream.guessed_rate)}

                    yield dict(video_d, frame=img, pts=frame.pts, video_meta=video_meta)


class AVClipReader(IterDataPipe[Dict[str, Any]]):
    def __init__(
            self,
            video_dp: IterDataPipe[Dict[str, Any]],
            num_frames_per_clip: int = 8,
            num_clips_per_video: int = 1,
            step_between_clips: int = 1) -> None:
        """TorchData Iterdatapype that takes in video datapipe
        and yields `num_clips_per_video` video clips (sequences of `num_frames_per_clip` frames) from a video.
        Clips are sampled from all possible clips of length `num_frames_per_clip` spaced `step_between_clips` apart.

        Args:
            video_dp (IterDataPipe[Dict[str, Any]]): Video dataset IterDataPipe
            num_frames_per_clip (int, optional): Length of a video clip in frames. Defaults to 8.
            num_clips_per_video (int, optional): How many clips to sample from each video. Defaults to 1.
            step_between_clips (int, optional): Minimum spacing between two clips. Defaults to 1.
        """

        self.datapipe = video_dp
        self.num_frames_per_clip = num_frames_per_clip
        self.num_clips_per_video = num_clips_per_video
        self.step_between_clips = step_between_clips

    def _unfold(self, tensor: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """
        similar to tensor.unfold, but with the dilation
        and specialized for 1d tensors
        Returns all consecutive windows of `self.num_frames_per_clip` elements, with
        `self.step_between_clips` between windows. The distance between each element
        in a window is given by `dilation`.
        """
        assert tensor.dim() == 1
        o_stride = tensor.stride(0)
        numel = tensor.numel()
        new_stride = (self.step_between_clips * o_stride, dilation * o_stride)
        new_size = ((numel - (dilation * (self.num_frames_per_clip - 1) + 1)) // self.step_between_clips + 1,
                    self.num_frames_per_clip)
        if new_size[0] < 1:
            new_size = (0, self.num_frames_per_clip)
        return torch.as_strided(tensor, new_size, new_stride)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for video_d in self.datapipe:
            buffer = video_d["file"]
            with av.open(buffer, metadata_errors="ignore") as container:
                stream = container.streams.video[0]
                time_base = stream.time_base

                # duration is given in time_base units as int
                duration = stream.duration

                # get video_stream timestramps
                # with a tolerance for pyav imprecission
                _ptss = torch.arange(duration - 7)
                _ptss = self._unfold(_ptss)
                # shuffle the clips
                perm = torch.randperm(_ptss.size(0))
                idx = perm[:self.num_clips_per_video]
                samples = _ptss[idx]

                for clip_pts in samples:
                    start_pts = clip_pts[0].item()
                    end_pts = clip_pts[-1].item()
                    # video_timebase is the default time_base
                    pts_unit = "pts"
                    start_pts, end_pts, pts_unit = _video_opt._convert_to_sec(start_pts, end_pts, "pts", time_base)
                    video_frames = video._read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        stream,
                        {"video": 0},
                    )

                    vframes_list = [frame.to_ndarray(format='rgb24') for frame in video_frames]

                    if vframes_list:
                        vframes = torch.as_tensor(np.stack(vframes_list))
                        # account for rounding errors in conversion
                        # FIXME: fix this in the code
                        vframes = vframes[:self.num_frames_per_clip, ...]

                    else:
                        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)
                        print("FAIL")

                    # [N,H,W,C] to [N,C,H,W]
                    vframes = vframes.permute(0, 3, 1, 2)
                    assert(vframes.size(0) == self.num_frames_per_clip)

                    # TODO: support sampling rates (FPS change)
                    # TODO: optimization (read all and select)

                    yield {
                        "clip": vframes,
                        "pts": clip_pts,
                        "range": (start_pts, end_pts),
                        "video_meta": {
                            "time_base": float(stream.time_base),
                            "guessed_fps": float(stream.guessed_rate),
                        },
                        "path": video_d["path"],
                        "target": video_d["target"]
                    }
