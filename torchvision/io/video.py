import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Optional, Union

import numpy as np
import torch

from ..utils import _log_api_usage_once
from ._video_deprecation_warning import _raise_video_deprecation_warning

try:
    import av

    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, "pict_type"):
        av = ImportError(
            """\
Your version of PyAV is too old for the necessary video operations in torchvision.
If you are on Python 3.5, you will have to build from source (the conda-forge
packages are not up-to-date).  See
https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
"""
        )
    try:
        FFmpegError = av.FFmpegError  # from av 14 https://github.com/PyAV-Org/PyAV/blob/main/CHANGELOG.rst
    except AttributeError:
        FFmpegError = av.AVError
except ImportError:
    av = ImportError(
        """\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
"""
    )


def _check_av_available() -> None:
    if isinstance(av, Exception):
        raise av





def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[dict[str, Any]] = None,
) -> None:
    """
    [DEPRECATED] Writes a 4d tensor in [T, H, W, C] format in a video file.

    .. warning::

        DEPRECATED: All the video decoding and encoding capabilities of torchvision
        are deprecated from version 0.22 and will be removed in version 0.25.  We
        recommend that you migrate to
        `TorchCodec <https://github.com/pytorch/torchcodec>`__, where we'll
        consolidate the future decoding/encoding capabilities of PyTorch

    This function relies on PyAV (therefore, ultimately FFmpeg) to encode
    videos, you can get more fine-grained control by referring to the other
    options at your disposal within `the FFMpeg wiki
    <http://trac.ffmpeg.org/wiki#Encoding>`_.

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream.
            The list of options is codec-dependent and can all
            be found from `the FFMpeg wiki <http://trac.ffmpeg.org/wiki#Encoding>`_.
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream.
            The list of options is codec-dependent and can all
            be found from `the FFMpeg wiki <http://trac.ffmpeg.org/wiki#Encoding>`_.

    Examples::
        >>> # Creating libx264 video with CRF 17, for visually lossless footage:
        >>>
        >>> from torchvision.io import write_video
        >>> # 1000 frames of 100x100, 3-channel image.
        >>> vid = torch.randn(1000, 100, 100, 3, dtype = torch.uint8)
        >>> write_video("video.mp4", options = {"crf": "17"})

    """
    _raise_video_deprecation_warning()
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(write_video)
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy(force=True)

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = int(np.round(fps))

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy(force=True).astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(audio_array, format=audio_sample_fmt, layout=audio_layout)

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            try:
                frame.pict_type = "NONE"
            except TypeError:
                from av.video.frame import PictureType  # noqa

                frame.pict_type = PictureType.NONE

            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

