from typing import Any, Dict, Iterator

import torch

from ..utils import _log_api_usage_once

try:
    from ._load_gpu_decoder import _HAS_VIDEO_DECODER
except ModuleNotFoundError:
    _HAS_VIDEO_DECODER = False
from ._video_opt import (
    Timebase,
    VideoMetaData,
    _HAS_VIDEO_OPT,
    _probe_video_from_file,
    _probe_video_from_memory,
    _read_video_from_file,
    _read_video_from_memory,
    _read_video_timestamps_from_file,
    _read_video_timestamps_from_memory,
)
from .image import (
    ImageReadMode,
    decode_image,
    decode_jpeg,
    decode_png,
    encode_jpeg,
    encode_png,
    read_file,
    read_image,
    write_file,
    write_jpeg,
    write_png,
)
from .video import (
    read_video,
    read_video_timestamps,
    write_video,
)


if _HAS_VIDEO_OPT:

    def _has_video_opt() -> bool:
        return True


else:

    def _has_video_opt() -> bool:
        return False


class VideoReader:
    """
    Fine-grained video-reading API.
    Supports frame-by-frame reading of various streams from a single video
    container.

    Example:
        The following examples creates a :mod:`VideoReader` object, seeks into 2s
        point, and returns a single frame::

            import torchvision
            video_path = "path_to_a_test_video"
            reader = torchvision.io.VideoReader(video_path, "video")
            reader.seek(2.0)
            frame = next(reader)

        :mod:`VideoReader` implements the iterable API, which makes it suitable to
        using it in conjunction with :mod:`itertools` for more advanced reading.
        As such, we can use a :mod:`VideoReader` instance inside for loops::

            reader.seek(2)
            for frame in reader:
                frames.append(frame['data'])
            # additionally, `seek` implements a fluent API, so we can do
            for frame in reader.seek(2):
                frames.append(frame['data'])

        With :mod:`itertools`, we can read all frames between 2 and 5 seconds with the
        following code::

            for frame in itertools.takewhile(lambda x: x['pts'] <= 5, reader.seek(2)):
                frames.append(frame['data'])

        and similarly, reading 10 frames after the 2s timestamp can be achieved
        as follows::

            for frame in itertools.islice(reader.seek(2), 10):
                frames.append(frame['data'])

    .. note::

        Each stream descriptor consists of two parts: stream type (e.g. 'video') and
        a unique stream id (which are determined by the video encoding).
        In this way, if the video contaner contains multiple
        streams of the same type, users can acces the one they want.
        If only stream type is passed, the decoder auto-detects first stream of that type.

    Args:

        path (string): Path to the video file in supported format

        stream (string, optional): descriptor of the required stream, followed by the stream id,
            in the format ``{stream_type}:{stream_id}``. Defaults to ``"video:0"``.
            Currently available options include ``['video', 'audio']``

        num_threads (int, optional): number of threads used by the codec to decode video.
            Default value (0) enables multithreading with codec-dependent heuristic. The performance
            will depend on the version of FFMPEG codecs supported.

        device (str, optional): Device to be used for decoding. Defaults to ``"cpu"``.

    """

    def __init__(self, path: str, stream: str = "video", num_threads: int = 0, device: str = "cpu") -> None:
        _log_api_usage_once(self)
        self.is_cuda = False
        device = torch.device(device)
        if device.type == "cuda":
            if not _HAS_VIDEO_DECODER:
                raise RuntimeError("Not compiled with GPU decoder support.")
            self.is_cuda = True
            if device.index is None:
                raise RuntimeError("Invalid cuda device!")
            self._c = torch.classes.torchvision.GPUDecoder(path, device.index)
            return
        if not _has_video_opt():
            raise RuntimeError(
                "Not compiled with video_reader support, "
                + "to enable video_reader support, please install "
                + "ffmpeg (version 4.2 is currently supported) and "
                + "build torchvision from source."
            )

        self._c = torch.classes.torchvision.Video(path, stream, num_threads)

    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        if self.is_cuda:
            frame = self._c.next()
            if frame.numel() == 0:
                raise StopIteration
            return {"data": frame}
        frame, pts = self._c.next()
        if frame.numel() == 0:
            raise StopIteration
        return {"data": frame, "pts": pts}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def seek(self, time_s: float, keyframes_only: bool = False) -> "VideoReader":
        """Seek within current stream.

        Args:
            time_s (float): seek time in seconds
            keyframes_only (bool): allow to seek only to keyframes

        .. note::
            Current implementation is the so-called precise seek. This
            means following seek, call to :mod:`next()` will return the
            frame with the exact timestamp if it exists or
            the first frame with timestamp larger than ``time_s``.
        """
        self._c.seek(time_s, keyframes_only)
        return self

    def get_metadata(self) -> Dict[str, Any]:
        """Returns video metadata

        Returns:
            (dict): dictionary containing duration and frame rate for every stream
        """
        return self._c.get_metadata()

    def set_current_stream(self, stream: str) -> bool:
        """Set current stream.
        Explicitly define the stream we are operating on.

        Args:
            stream (string): descriptor of the required stream. Defaults to ``"video:0"``
                Currently available stream types include ``['video', 'audio']``.
                Each descriptor consists of two parts: stream type (e.g. 'video') and
                a unique stream id (which are determined by video encoding).
                In this way, if the video contaner contains multiple
                streams of the same type, users can acces the one they want.
                If only stream type is passed, the decoder auto-detects first stream
                of that type and returns it.

        Returns:
            (bool): True on succes, False otherwise
        """
        if self.is_cuda:
            print("GPU decoding only works with video stream.")
        return self._c.set_current_stream(stream)


__all__ = [
    "write_video",
    "read_video",
    "read_video_timestamps",
    "_read_video_from_file",
    "_read_video_timestamps_from_file",
    "_probe_video_from_file",
    "_read_video_from_memory",
    "_read_video_timestamps_from_memory",
    "_probe_video_from_memory",
    "_HAS_VIDEO_OPT",
    "_HAS_VIDEO_DECODER",
    "_read_video_clip_from_memory",
    "_read_video_meta_data",
    "VideoMetaData",
    "Timebase",
    "ImageReadMode",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "encode_jpeg",
    "encode_png",
    "read_file",
    "read_image",
    "write_file",
    "write_jpeg",
    "write_png",
    "Video",
]
