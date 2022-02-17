import random
import warnings
from typing import Any, Dict, Iterator, Optional, Tuple

import av
import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchvision import get_video_backend
from torchvision.io import video, _video_opt, VideoReader
from torchvision.prototype.features import Image, EncodedVideo
from torchvision.prototype.utils._internal import ReadOnlyTensorBuffer, query_recursively


class _VideoDecoder(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, *, inline: bool = True) -> None:
        # TODO: add gpu support
        self.datapipe = datapipe
        self._inline = inline

    def _decode(self, buffer: ReadOnlyTensorBuffer, meta: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def _find_encoded_video(self, id: Tuple[Any, ...], obj: Any) -> Optional[Tuple[Any, ...]]:
        if isinstance(obj, EncodedVideo):
            return id, obj
        else:
            return None

    def _integrate_data(self, sample: Any, id: Tuple[Any, ...], data: Dict[str, Any]) -> Any:
        if not self._inline:
            return sample, data
        elif not id:
            return data

        grand_parent = None
        parent = sample
        for item in id[:-1]:
            grand_parent = parent
            parent = parent[item]

        if not isinstance(parent, dict):
            raise TypeError(
                f"Could not inline the decoded video data, "
                f"since the container at item {''.join(str([item]) for item in id[:-1])} "
                f"that holds the `EncodedVideo` at item {[id[-1]]} is not a 'dict' but a '{type(parent).__name__}'. "
                f"If you don't want to automatically inline the decoded video data, construct the decoder with "
                f"{type(self).__name__}(..., inline=False). This will change the return type to a tuple of the input "
                f"and the decoded video data for each iteration."
            )

        parent = parent.copy()
        del parent[id[-1]]
        parent.update(data)

        if not grand_parent:
            return parent

        grand_parent[id[-2]] = parent
        return sample

    def __iter__(self) -> Iterator[Any]:
        for sample in self.datapipe:
            ids_and_videos = list(query_recursively(self._find_encoded_video, sample))
            if not ids_and_videos:
                raise TypeError("no encoded video")
            elif len(ids_and_videos) > 1:
                raise ValueError("more than one encoded video")
            id, video = ids_and_videos[0]

            for data in self._decode(ReadOnlyTensorBuffer(video), video.meta.copy()):
                yield self._integrate_data(sample, id, data)


class KeyframeDecoder(_VideoDecoder):
    def _decode(self, buffer: ReadOnlyTensorBuffer, meta: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if get_video_backend() == "video_reader":
            warnings.warn("Video reader API not implemented for keyframes yet, reverting to PyAV")
            
        with av.open(buffer, metadata_errors="ignore") as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            for frame in container.decode(stream):
                yield dict(
                    frame=Image.from_pil(frame.to_image()),
                    pts=frame.pts,
                    video_meta=dict(
                        meta,
                        time_base=float(frame.time_base),
                        guessed_fps=float(stream.guessed_rate),
                    ),
                )


class RandomFrameDecoder(_VideoDecoder):
    def __init__(self, datapipe: IterDataPipe, *, num_samples: int = 1, inline: bool = True) -> None:
        super().__init__(datapipe, inline=inline)
        self.num_sampler = num_samples

    def _decode(self, buffer: ReadOnlyTensorBuffer, meta: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if get_video_backend() == "video_reader":
            vid = VideoReader(buffer, device=self.device)
            # seek and return frames
            metadata = vid.get_metadata()["video"]
            duration = metadata["duration"][0] if self.device == "cpu" else metadata["duration"]
            fps = metadata["fps"][0] if self.device == "cpu" else metadata["fps"]
            max_seek = duration - (self.clip_len / fps + 0.1)  # FIXME: random param
            seek_idxs = random.sample(list(range(max_seek)), self.num_samples)
            for i in seek_idxs:
                vid.seek(i)
                frame = vid.next()
                yield dict(
                    frame=frame['data'],
                    pts = frame['pts'],
                    video_meta=dict(
                        guessed_fps=fps,
                    ),
                )
        else:
            with av.open(buffer, metadata_errors="ignore") as container:
                stream = container.streams.video[0]
                # duration is given in time_base units as int
                duration = stream.duration
                # seek to a random frame
                seek_idxs = random.sample(list(range(duration)), self.num_samples)
                for i in seek_idxs:
                    container.seek(i, any_frame=True, stream=stream)
                    frame = next(container.decode(stream))
                    yield dict(
                        frame=Image.from_pil(frame.to_image()),
                        pts=frame.pts,
                        video_meta=dict(
                            time_base=float(frame.time_base),
                            guessed_fps=float(stream.guessed_rate),
                        ),
                    )

class ClipDecoder(_VideoDecoder):
    def __init__(
        self,
        datapipe: IterDataPipe,
        *,
        num_frames_per_clip: int = 8,
        num_clips_per_video: int = 1,
        step_between_clips: int = 1,
        inline: bool = True,
    ) -> None:
        super().__init__(datapipe, inline=inline)
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
        new_size = (
            (numel - (dilation * (self.num_frames_per_clip - 1) + 1)) // self.step_between_clips + 1,
            self.num_frames_per_clip,
        )
        if new_size[0] < 1:
            new_size = (0, self.num_frames_per_clip)
        return torch.as_strided(tensor, new_size, new_stride)

    def _decode(self, buffer: ReadOnlyTensorBuffer, meta: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if get_video_backend() == "video_reader":
            pass
        else:
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
                idx = perm[: self.num_clips_per_video]
                samples = _ptss[idx]

                for clip_pts in samples:
                    start_pts = clip_pts[0].item()
                    end_pts = clip_pts[-1].item()
                    # video_timebase is the default time_base
                    pts_unit = "pts"
                    start_pts, end_pts, pts_unit = _video_opt._convert_to_sec(start_pts, end_pts, "pts", time_base)
                    video_frames = video._read_from_stream(
                        container,
                        float(start_pts),
                        float(end_pts),
                        pts_unit,
                        stream,
                        {"video": 0},
                    )

                    vframes_list = [frame.to_ndarray(format="rgb24") for frame in video_frames]

                    if vframes_list:
                        vframes = torch.as_tensor(np.stack(vframes_list))
                        # account for rounding errors in conversion
                        # FIXME: fix this in the code
                        vframes = vframes[: self.num_frames_per_clip, ...]

                    else:
                        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)
                        print("FAIL")

                    # [N,H,W,C] to [N,C,H,W]
                    vframes = vframes.permute(0, 3, 1, 2)
                    assert vframes.size(0) == self.num_frames_per_clip

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
                    }
