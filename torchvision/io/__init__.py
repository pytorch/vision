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
    Video = torch.classes.torchvision.Video
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
