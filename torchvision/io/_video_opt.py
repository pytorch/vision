from fractions import Fraction
import numpy as np
import os
import torch
import imp
import warnings


_HAS_VIDEO_OPT = False

try:
    lib_dir = os.path.join(os.path.dirname(__file__), '..')
    _, path, description = imp.find_module("video_reader", [lib_dir])
    torch.ops.load_library(path)
    _HAS_VIDEO_OPT = True
except (ImportError, OSError):
    warnings.warn("video reader based on ffmpeg c++ ops not available")

default_timebase = Fraction(0, 1)


def _validate_pts(pts_range):
    if pts_range[1] > 0:
        assert pts_range[0] <= pts_range[1], \
            """Start pts should not be smaller than end pts, got
            start pts: %d and end pts: %d""" % (pts_range[0], pts_range[1])


def _fill_info(vtimebase, vfps, atimebase, asample_rate):
    info = {}
    if vtimebase.numel() > 0:
        info["video_timebase"] = Fraction(vtimebase[0].item(), vtimebase[1].item())
    if vfps.numel() > 0:
        info["video_fps"] = vfps.item()
    if atimebase.numel() > 0:
        info["audio_timebase"] = Fraction(atimebase[0].item(), atimebase[1].item())
    if asample_rate.numel() > 0:
        info["audio_sample_rate"] = asample_rate.item()

    return info


def _align_audio_frames(aframes, aframe_pts, audio_pts_range):
    start, end = aframe_pts[0], aframe_pts[-1]
    num_samples = aframes.size(0)
    step_per_aframe = float(end - start + 1) / float(num_samples)
    s_idx = 0
    e_idx = num_samples
    if start < audio_pts_range[0]:
        s_idx = int((audio_pts_range[0] - start) / step_per_aframe)
    if end > audio_pts_range[1]:
        e_idx = int((audio_pts_range[1] - end) / step_per_aframe)
    return aframes[s_idx:e_idx, :]


def _read_video_from_file(
    filename,
    seek_frame_margin=0.25,
    read_video_stream=True,
    video_width=0,
    video_height=0,
    video_min_dimension=0,
    video_pts_range=(0, -1),
    video_timebase=default_timebase,
    read_audio_stream=True,
    audio_samples=0,
    audio_channels=0,
    audio_pts_range=(0, -1),
    audio_timebase=default_timebase,
):
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames

    Args
    ----------
    filename : str
        path to the video file
    seek_frame_margin: double, optional
        seeking frame in the stream is imprecise. Thus, when video_start_pts is specified,
        we seek the pts earlier by seek_frame_margin seconds
    read_video_stream: int, optional
        whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension: int
        together decide the size of decoded frames
        - when video_width = 0, video_height = 0, and video_min_dimension = 0, keep the orignal frame resolution
        - when video_width = 0, video_height = 0, and video_min_dimension != 0, keep the aspect ratio and resize
            the frame so that shorter edge size is video_min_dimension
        - When video_width = 0, and video_height != 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_height
        - When video_width != 0, and video_height == 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_width
        - When video_width != 0, and video_height != 0, resize the frame so that frame video_width and video_height
            are set to $video_width and $video_height, respectively
    video_pts_range : list(int), optional
        the start and end presentation timestamp of video stream
    video_timebase: Fraction, optional
        a Fraction rational number which denotes timebase in video stream
    read_audio_stream: int, optional
        whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples: int, optional
        audio sampling rate
    audio_channels: int optional
        audio channels
    audio_pts_range : list(int), optional
        the start and end presentation timestamp of audio stream
    audio_timebase: Fraction, optional
        a Fraction rational number which denotes time base in audio stream

    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[L, K]
        the audio frames, where `L` is the number of points and
            `K` is the number of audio_channels
    info : Dict
        metadata for the video and audio. Can contain the fields video_fps (float)
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
    vframes, _vframe_pts, vtimebase, vfps, aframes, aframe_pts, atimebase, asample_rate = result
    info = _fill_info(vtimebase, vfps, atimebase, asample_rate)
    if aframes.numel() > 0:
        # when audio stream is found
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return vframes, aframes, info


def _read_video_timestamps_from_file(filename):
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
    _vframes, vframe_pts, vtimebase, vfps, _aframes, aframe_pts, atimebase, asample_rate = result
    info = _fill_info(vtimebase, vfps, atimebase, asample_rate)

    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return vframe_pts, aframe_pts, info


def _read_video_from_memory(
    file_buffer,
    seek_frame_margin=0.25,
    read_video_stream=1,
    video_width=0,
    video_height=0,
    video_min_dimension=0,
    video_pts_range=(0, -1),
    video_timebase=default_timebase,
    read_audio_stream=1,
    audio_samples=0,
    audio_channels=0,
    audio_pts_range=(0, -1),
    audio_timebase=default_timebase,
):
    """
    Reads a video from memory, returning both the video frames as well as
    the audio frames

    Args
    ----------
    file_buffer : buffer
        buffer of compressed video content
    seek_frame_margin: double, optional
        seeking frame in the stream is imprecise. Thus, when video_start_pts is specified,
        we seek the pts earlier by seek_frame_margin seconds
    read_video_stream: int, optional
        whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension: int
        together decide the size of decoded frames
        - when video_width = 0, video_height = 0, and video_min_dimension = 0, keep the orignal frame resolution
        - when video_width = 0, video_height = 0, and video_min_dimension != 0, keep the aspect ratio and resize
            the frame so that shorter edge size is video_min_dimension
        - When video_width = 0, and video_height != 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_height
        - When video_width != 0, and video_height == 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_width
        - When video_width != 0, and video_height != 0, resize the frame so that frame video_width and video_height
            are set to $video_width and $video_height, respectively
    video_pts_range : list(int), optional
        the start and end presentation timestamp of video stream
    video_timebase: Fraction, optional
        a Fraction rational number which denotes timebase in video stream
    read_audio_stream: int, optional
        whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples: int, optional
        audio sampling rate
    audio_channels: int optional
        audio audio_channels
    audio_pts_range : list(int), optional
        the start and end presentation timestamp of audio stream
    audio_timebase: Fraction, optional
        a Fraction rational number which denotes time base in audio stream

    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[L, K]
        the audio frames, where `L` is the number of points and
            `K` is the number of channels
    info : Dict
        metadata for the video and audio. Can contain the fields video fps (float)
        and audio sample rate (int)
    """

    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)

    video_tensor = torch.from_numpy(np.frombuffer(file_buffer, dtype=np.uint8))

    result = torch.ops.video_reader.read_video_from_memory(
        video_tensor,
        seek_frame_margin,
        0,  # getPtsOnly
        read_video_stream,
        video_width,
        video_height,
        video_min_dimension,
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

    vframes, _vframe_pts, vtimebase, vfps, aframes, aframe_pts, atimebase, asample_rate = result
    info = _fill_info(vtimebase, vfps, atimebase, asample_rate)
    if aframes.numel() > 0:
        # when audio stream is found
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return vframes, aframes, info


def _read_video_timestamps_from_memory(file_buffer):
    """
    Decode all frames in the video. Only pts (presentation timestamp) is returned.
    The actual frame pixel data is not copied. Thus, read_video_timestamps(...)
    is much faster than read_video(...)
    """

    video_tensor = torch.from_numpy(np.frombuffer(file_buffer, dtype=np.uint8))
    result = torch.ops.video_reader.read_video_from_memory(
        video_tensor,
        0,  # seek_frame_margin
        1,  # getPtsOnly
        1,  # read_video_stream
        0,  # video_width
        0,  # video_height
        0,  # video_min_dimension
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
    _vframes, vframe_pts, vtimebase, vfps, _aframes, aframe_pts, atimebase, asample_rate = result
    info = _fill_info(vtimebase, vfps, atimebase, asample_rate)

    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return vframe_pts, aframe_pts, info
