import torch
import warnings

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
from .video import (
    read_video,
    read_video_timestamps,
    write_video,
)
from .image import (
    read_image,
    decode_image,
    encode_jpeg,
    write_jpeg,
    encode_png,
    write_png,
)


if _HAS_VIDEO_OPT:

    class Video:
        """
        Fine-grained video-reading API.
        Supports frame-by-frame reading of various streams from a single video
        container.

        Args:

            path (string): Path to the video file in supported format

            stream (string, optional): descriptor of the required stream. Defaults to "video:0"
                Currently available options include :mod:`['video', 'audio', 'cc', 'sub']`

        Example:
            The following examples creates :mod:`Video` object, seeks into 2s
            point, and returns a single frame::
                    import torchvision
                    video_path = "path_to_a_test_video"

                    reader = torchvision.io.Video(video_path, "video")
                    reader.seek(2.0)
                    frame, timestamp = reader.next()
        """

        def __init__(self, path, stream="video"):
            self._c = torch.classes.torchvision.Video(path, stream)

        def next(self):
            """Iterator that decodes the next frame of the current stream

            Returns:
                ([torch.Tensor, float]): list containing decoded frame and corresponding timestamp

            """
            return self._c.next()

        def seek(self, time_s: float):
            """Seek within current stream.

            Args:
                time_s (float): seek time in seconds

            .. note::
                Current implementation is the so-called precise seek. This
                means following seek, call to :mod:`next()` will return the
                frame with the exact timestamp if it exists or
                the first frame with timestamp larger than time_s.
            """
            self._c.seek(time_s)

        def get_metadata(self):
            """Returns video metadata

            Returns:
                (dict): dictionary containing duration and frame rate for every stream
            """
            return self._c.get_metadata()

        def set_current_stream(self, stream: str):
            """Set current stream.
            Explicitly define the stream we are operating on.

            Args:
                stream (string): descriptor of the required stream. Defaults to "video:0"
                    Currently available stream types include :mod:`['video', 'audio', 'cc', 'sub']`.
                    Each descriptor consists of two parts: stream type (e.g. 'video') and
                    a unique stream id (which are determined by video encoding).
                    In this way, if the video contaner contains multiple
                    streams of the same type, users can acces the one they want.
                    If only stream type is passed, the decoder auto-detects first stream
                    of that type and returns it.

            Returns:
                (bool): True on succes, False otherwise
            """
            return self._c.set_current_stream(stream)


else:
    Video = None


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
    "_read_video_clip_from_memory",
    "_read_video_meta_data",
    "VideoMetaData",
    "Timebase",
    "read_image",
    "decode_image",
    "encode_jpeg",
    "write_jpeg",
    "encode_png",
    "write_png",
    "Video",
]
