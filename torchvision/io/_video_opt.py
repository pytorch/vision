
import math
import os
import warnings
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Type, cast

import numpy as np
import torch

from .._internally_replaced_utils import _get_extension_path


try:
    lib_path = _get_extension_path('video_reader')
    torch.ops.load_library(lib_path)
    _HAS_VIDEO_OPT = True
except (ImportError, OSError):
    _HAS_VIDEO_OPT = False

default_timebase = Fraction(0, 1)


# simple class for torch scripting
# the complex Fraction class from fractions module is not scriptable
class Timebase(object):
    __annotations__: Dict[str, Type[int]] = {"numerator": int, "denominator": int}
    __slots__: List[str] = ["numerator", "denominator"]

    def __init__(
        self,
        numerator: int,
        denominator: int,
    ) -> None:
        self.numerator = numerator
        self.denominator = denominator


class VideoMetaData(object):
    __annotations__: Dict[str, Any] = {
        "has_video": bool,
        "video_timebase": Timebase,
        "video_duration": float,
        "video_fps": float,
        "has_audio": bool,
        "audio_timebase": Timebase,
        "audio_duration": float,
        "audio_sample_rate": float,
    }
    __slots__: List[str] = [
        "has_video",
        "video_timebase",
        "video_duration",
        "video_fps",
        "has_audio",
        "audio_timebase",
        "audio_duration",
        "audio_sample_rate",
    ]

    def __init__(self) -> None:
        self.has_video = False
        self.video_timebase = Timebase(0, 1)
        self.video_duration = 0.0
        self.video_fps = 0.0
        self.has_audio = False
        self.audio_timebase = Timebase(0, 1)
        self.audio_duration = 0.0
        self.audio_sample_rate = 0.0


def _validate_pts(pts_range: Tuple[int, int]) -> None:

    if pts_range[1] > 0:
        assert (
            pts_range[0] <= pts_range[1]
        ), """Start pts should not be smaller than end pts, got
            start pts: {0:d} and end pts: {1:d}""".format(
            pts_range[0],
            pts_range[1],
        )


def _fill_info(
    vtimebase: torch.Tensor,
    vfps: torch.Tensor,
    vduration: torch.Tensor,
    atimebase: torch.Tensor,
    asample_rate: torch.Tensor,
    aduration: torch.Tensor,
) -> VideoMetaData:
    """
    Build update VideoMetaData struct with info about the video
    """
    meta = VideoMetaData()
    if vtimebase.numel() > 0:
        meta.video_timebase = Timebase(
            int(vtimebase[0].item()), int(vtimebase[1].item())
        )
        timebase = vtimebase[0].item() / float(vtimebase[1].item())
        if vduration.numel() > 0:
            meta.has_video = True
            meta.video_duration = float(vduration.item()) * timebase
    if vfps.numel() > 0:
        meta.video_fps = float(vfps.item())
    if atimebase.numel() > 0:
        meta.audio_timebase = Timebase(
            int(atimebase[0].item()), int(atimebase[1].item())
        )
        timebase = atimebase[0].item() / float(atimebase[1].item())
        if aduration.numel() > 0:
            meta.has_audio = True
            meta.audio_duration = float(aduration.item()) * timebase
    if asample_rate.numel() > 0:
        meta.audio_sample_rate = float(asample_rate.item())

    return meta


def _align_audio_frames(
    aframes: torch.Tensor,
    aframe_pts: torch.Tensor,
    audio_pts_range: Tuple[int, int]
) -> torch.Tensor:
    start, end = aframe_pts[0], aframe_pts[-1]
    num_samples = aframes.size(0)
    step_per_aframe = float(end - start + 1) / float(num_samples)
    s_idx = 0
    e_idx = num_samples
    if start < audio_pts_range[0]:
        s_idx = int((audio_pts_range[0] - start) / step_per_aframe)
    if audio_pts_range[1] != -1 and end > audio_pts_range[1]:
        e_idx = int((audio_pts_range[1] - end) / step_per_aframe)
    return aframes[s_idx:e_idx, :]


def _read_video_from_file(
    filename: str,
    seek_frame_margin: float = 0.25,
    read_video_stream: bool = True,
    video_width: int = 0,
    video_height: int = 0,
    video_min_dimension: int = 0,
    video_max_dimension: int = 0,
    video_pts_range: Tuple[int, int] = (0, -1),
    video_timebase: Fraction = default_timebase,
    read_audio_stream: bool = True,
    audio_samples: int = 0,
    audio_channels: int = 0,
    audio_pts_range: Tuple[int, int] = (0, -1),
    audio_timebase: Fraction = default_timebase,
) -> Tuple[torch.Tensor, torch.Tensor, VideoMetaData]:
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames

    Args:
    filename (str): path to the video file
    seek_frame_margin (double, optional): seeking frame in the stream is imprecise. Thus,
        when video_start_pts is specified, we seek the pts earlier by seek_frame_margin seconds
    read_video_stream (int, optional): whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension/video_max_dimension (int): together decide
        the size of decoded frames:

            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the original frame resolution
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension = 0, keep the aspect ratio and resize the
                frame so that shorter edge size is video_min_dimension
            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension != 0, keep the aspect ratio and resize
                the frame so that longer edge size is video_max_dimension
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension != 0, resize the frame so that shorter
                edge size is video_min_dimension, and longer edge size is
                video_max_dimension. The aspect ratio may not be preserved
            - When video_width = 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_height is $video_height
            - When video_width != 0, video_height == 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_width is $video_width
            - When video_width != 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, resize the frame so that frame
                video_width and  video_height are set to $video_width and
                $video_height, respectively
    video_pts_range (list(int), optional): the start and end presentation timestamp of video stream
    video_timebase (Fraction, optional): a Fraction rational number which denotes timebase in video stream
    read_audio_stream (int, optional): whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples (int, optional): audio sampling rate
    audio_channels (int optional): audio channels
    audio_pts_range (list(int), optional): the start and end presentation timestamp of audio stream
    audio_timebase (Fraction, optional): a Fraction rational number which denotes time base in audio stream

    Returns
        vframes (Tensor[T, H, W, C]): the `T` video frames
        aframes (Tensor[L, K]): the audio frames, where `L` is the number of points and
            `K` is the number of audio_channels
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float)
            and audio_fps (int)
    """
    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)

    result = torch.ops.video_reader.read_video_from_file(
        filename,
        seek_frame_margin,
        0,  # getPtsOnly
        read_video_stream,
        video_width,
        video_height,
        video_min_dimension,
        video_max_dimension,
        video_pts_range[0],
        video_pts_range[1],
        video_timebase.numerator,
        video_timebase.denominator,
        read_audio_stream,
        audio_samples,
        audio_channels,
        audio_pts_range[0],
        audio_pts_range[1],
        audio_timebase.numerator,
        audio_timebase.denominator,
    )
    vframes, _vframe_pts, vtimebase, vfps, vduration, \
        aframes, aframe_pts, atimebase, asample_rate, aduration = (
            result
        )
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    if aframes.numel() > 0:
        # when audio stream is found
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return vframes, aframes, info


def _read_video_timestamps_from_file(filename: str) -> Tuple[List[int], List[int], VideoMetaData]:
    """
    Decode all video- and audio frames in the video. Only pts
    (presentation timestamp) is returned. The actual frame pixel data is not
    copied. Thus, it is much faster than read_video(...)
    """
    result = torch.ops.video_reader.read_video_from_file(
        filename,
        0,  # seek_frame_margin
        1,  # getPtsOnly
        1,  # read_video_stream
        0,  # video_width
        0,  # video_height
        0,  # video_min_dimension
        0,  # video_max_dimension
        0,  # video_start_pts
        -1,  # video_end_pts
        0,  # video_timebase_num
        1,  # video_timebase_den
        1,  # read_audio_stream
        0,  # audio_samples
        0,  # audio_channels
        0,  # audio_start_pts
        -1,  # audio_end_pts
        0,  # audio_timebase_num
        1,  # audio_timebase_den
    )
    _vframes, vframe_pts, vtimebase, vfps, vduration, \
        _aframes, aframe_pts, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)

    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return vframe_pts, aframe_pts, info


def _probe_video_from_file(filename: str) -> VideoMetaData:
    """
    Probe a video file and return VideoMetaData with info about the video
    """
    result = torch.ops.video_reader.probe_video_from_file(filename)
    vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    return info


def _read_video_from_memory(
    video_data: torch.Tensor,
    seek_frame_margin: float = 0.25,
    read_video_stream: int = 1,
    video_width: int = 0,
    video_height: int = 0,
    video_min_dimension: int = 0,
    video_max_dimension: int = 0,
    video_pts_range: Tuple[int, int] = (0, -1),
    video_timebase_numerator: int = 0,
    video_timebase_denominator: int = 1,
    read_audio_stream: int = 1,
    audio_samples: int = 0,
    audio_channels: int = 0,
    audio_pts_range: Tuple[int, int] = (0, -1),
    audio_timebase_numerator: int = 0,
    audio_timebase_denominator: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads a video from memory, returning both the video frames as well as
    the audio frames
    This function is torchscriptable.

    Args:
    video_data (data type could be 1) torch.Tensor, dtype=torch.int8 or 2) python bytes):
        compressed video content stored in either 1) torch.Tensor 2) python bytes
    seek_frame_margin (double, optional): seeking frame in the stream is imprecise.
        Thus, when video_start_pts is specified, we seek the pts earlier by seek_frame_margin seconds
    read_video_stream (int, optional): whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension/video_max_dimension (int): together decide
        the size of decoded frames:

            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the original frame resolution
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension = 0, keep the aspect ratio and resize the
                frame so that shorter edge size is video_min_dimension
            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension != 0, keep the aspect ratio and resize
                the frame so that longer edge size is video_max_dimension
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension != 0, resize the frame so that shorter
                edge size is video_min_dimension, and longer edge size is
                video_max_dimension. The aspect ratio may not be preserved
            - When video_width = 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_height is $video_height
            - When video_width != 0, video_height == 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_width is $video_width
            - When video_width != 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, resize the frame so that frame
                video_width and  video_height are set to $video_width and
                $video_height, respectively
    video_pts_range (list(int), optional): the start and end presentation timestamp of video stream
    video_timebase_numerator / video_timebase_denominator (float, optional): a rational
        number which denotes timebase in video stream
    read_audio_stream (int, optional): whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples (int, optional): audio sampling rate
    audio_channels (int optional): audio audio_channels
    audio_pts_range (list(int), optional): the start and end presentation timestamp of audio stream
    audio_timebase_numerator / audio_timebase_denominator (float, optional):
        a rational number which denotes time base in audio stream

    Returns:
        vframes (Tensor[T, H, W, C]): the `T` video frames
        aframes (Tensor[L, K]): the audio frames, where `L` is the number of points and
            `K` is the number of channels
    """

    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)

    if not isinstance(video_data, torch.Tensor):
        video_data = torch.from_numpy(np.frombuffer(video_data, dtype=np.uint8))

    result = torch.ops.video_reader.read_video_from_memory(
        video_data,
        seek_frame_margin,
        0,  # getPtsOnly
        read_video_stream,
        video_width,
        video_height,
        video_min_dimension,
        video_max_dimension,
        video_pts_range[0],
        video_pts_range[1],
        video_timebase_numerator,
        video_timebase_denominator,
        read_audio_stream,
        audio_samples,
        audio_channels,
        audio_pts_range[0],
        audio_pts_range[1],
        audio_timebase_numerator,
        audio_timebase_denominator,
    )

    vframes, _vframe_pts, vtimebase, vfps, vduration, \
        aframes, aframe_pts, atimebase, asample_rate, aduration = (
            result
        )

    if aframes.numel() > 0:
        # when audio stream is found
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)

    return vframes, aframes


def _read_video_timestamps_from_memory(
    video_data: torch.Tensor,
) -> Tuple[List[int], List[int], VideoMetaData]:
    """
    Decode all frames in the video. Only pts (presentation timestamp) is returned.
    The actual frame pixel data is not copied. Thus, read_video_timestamps(...)
    is much faster than read_video(...)
    """
    if not isinstance(video_data, torch.Tensor):
        video_data = torch.from_numpy(np.frombuffer(video_data, dtype=np.uint8))
    result = torch.ops.video_reader.read_video_from_memory(
        video_data,
        0,  # seek_frame_margin
        1,  # getPtsOnly
        1,  # read_video_stream
        0,  # video_width
        0,  # video_height
        0,  # video_min_dimension
        0,  # video_max_dimension
        0,  # video_start_pts
        -1,  # video_end_pts
        0,  # video_timebase_num
        1,  # video_timebase_den
        1,  # read_audio_stream
        0,  # audio_samples
        0,  # audio_channels
        0,  # audio_start_pts
        -1,  # audio_end_pts
        0,  # audio_timebase_num
        1,  # audio_timebase_den
    )
    _vframes, vframe_pts, vtimebase, vfps, vduration, \
        _aframes, aframe_pts, atimebase, asample_rate, aduration = (
            result
        )
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)

    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return vframe_pts, aframe_pts, info


def _probe_video_from_memory(
    video_data: torch.Tensor,
) -> VideoMetaData:
    """
    Probe a video in memory and return VideoMetaData with info about the video
    This function is torchscriptable
    """
    if not isinstance(video_data, torch.Tensor):
        video_data = torch.from_numpy(np.frombuffer(video_data, dtype=np.uint8))
    result = torch.ops.video_reader.probe_video_from_memory(video_data)
    vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    return info


def _convert_to_sec(start_pts: float, end_pts: float, pts_unit: str, time_base: Fraction) -> Tuple[float, float, str]:
    if pts_unit == 'pts':
        start_pts = float(start_pts * time_base)
        end_pts = float(end_pts * time_base)
        pts_unit = 'sec'
    return start_pts, end_pts, pts_unit


def _read_video(
    filename: str,
    start_pts: int = 0,
    end_pts: Optional[float] = None,
    pts_unit: str = "pts"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if end_pts is None:
        end_pts = float("inf")

    if pts_unit == "pts":
        warnings.warn(
            "The pts_unit 'pts' gives wrong results and will be removed in a "
            + "follow-up version. Please use pts_unit 'sec'."
        )

    info = _probe_video_from_file(filename)

    has_video = info.has_video
    has_audio = info.has_audio
    video_pts_range = (0, -1)
    video_timebase = default_timebase
    audio_pts_range = (0, -1)
    audio_timebase = default_timebase
    time_base = default_timebase

    if has_video:
        video_timebase = Fraction(
            info.video_timebase.numerator, info.video_timebase.denominator
        )
        time_base = video_timebase

    if has_audio:
        audio_timebase = Fraction(
            info.audio_timebase.numerator, info.audio_timebase.denominator
        )
        time_base = time_base if time_base else audio_timebase

    # video_timebase is the default time_base
    start_pts_sec, end_pts_sec, pts_unit = _convert_to_sec(
        start_pts, end_pts, pts_unit, time_base)

    def get_pts(time_base):
        start_offset = start_pts_sec
        end_offset = end_pts_sec
        if pts_unit == "sec":
            start_offset = int(math.floor(start_pts_sec * (1 / time_base)))
            if end_offset != float("inf"):
                end_offset = int(math.ceil(end_pts_sec * (1 / time_base)))
        if end_offset == float("inf"):
            end_offset = -1
        return start_offset, end_offset

    if has_video:
        video_pts_range = get_pts(video_timebase)

    if has_audio:
        audio_pts_range = get_pts(audio_timebase)

    vframes, aframes, info = _read_video_from_file(
        filename,
        read_video_stream=True,
        video_pts_range=video_pts_range,
        video_timebase=video_timebase,
        read_audio_stream=True,
        audio_pts_range=audio_pts_range,
        audio_timebase=audio_timebase,
    )
    _info = {}
    if has_video:
        _info["video_fps"] = info.video_fps
    if has_audio:
        _info["audio_fps"] = info.audio_sample_rate

    return vframes, aframes, _info


def _read_video_timestamps(filename: str, pts_unit: str = "pts") -> Tuple[List[Fraction], Optional[float]]:
    if pts_unit == "pts":
        warnings.warn(
            "The pts_unit 'pts' gives wrong results and will be removed in a "
            + "follow-up version. Please use pts_unit 'sec'."
        )

    pts, _, info = _read_video_timestamps_from_file(filename)

    if pts_unit == "sec":
        video_time_base = Fraction(
            info.video_timebase.numerator, info.video_timebase.denominator
        )
        pts = [x * video_time_base for x in pts]  # type: ignore[misc]

    video_fps = info.video_fps if info.has_video else None

    return pts, video_fps  # type: ignore[return-value]
