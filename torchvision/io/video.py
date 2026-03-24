# This module re-exports video utilities from the internal fb location.
# The actual implementation lives in pytorch.vision.fb.io.video
from pytorch.vision.fb.io.video import (  # type: ignore[import-not-found]
    _align_audio_frames,
    _av_available,
    _check_av_available,
    _read_from_stream,
    av,
    read_video,
    read_video_timestamps,
    write_video,
)

__all__ = [
    "read_video",
    "read_video_timestamps",
    "write_video",
    "_read_from_stream",
    "_align_audio_frames",
    "_check_av_available",
    "_av_available",
    "av",
]
