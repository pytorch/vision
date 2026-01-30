# In fbcode, import from the fb-only location
# For OSS, these imports would fail (video_reader not available)
try:
    from pytorch.vision.fb.io import (  # type: ignore[import-not-found]
        _HAS_CPU_VIDEO_DECODER,
        _HAS_VIDEO_OPT,
        _probe_video_from_file,
        _probe_video_from_memory,
        _read_video_from_file,
        _read_video_from_memory,
        _read_video_timestamps_from_file,
        _read_video_timestamps_from_memory,
        _video_opt,
        Timebase,
        VideoMetaData,
        VideoReader,
    )
except ImportError:
    # OSS fallback - video_reader backend not available
    _HAS_CPU_VIDEO_DECODER = False
    _HAS_VIDEO_OPT = False

    def _stub_not_available(*args, **kwargs):
        raise RuntimeError(
            "video_reader backend is not available in open-source torchvision. " "Use PyAV or TorchCodec instead."
        )

    _probe_video_from_file = _stub_not_available
    _probe_video_from_memory = _stub_not_available
    _read_video_from_file = _stub_not_available
    _read_video_from_memory = _stub_not_available
    _read_video_timestamps_from_file = _stub_not_available
    _read_video_timestamps_from_memory = _stub_not_available

    class Timebase:  # type: ignore[no-redef]
        __annotations__ = {"numerator": int, "denominator": int}
        __slots__ = ["numerator", "denominator"]

        def __init__(self, numerator: int = 0, denominator: int = 1) -> None:
            self.numerator = numerator
            self.denominator = denominator

    class VideoMetaData:  # type: ignore[no-redef]
        pass

    class VideoReader:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "VideoReader with video_reader backend is not available. "
                "Use backend='pyav' or migrate to TorchCodec."
            )

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    # Stub module for _video_opt to prevent circular import issues
    # This module is imported by video.py
    import types
    from fractions import Fraction

    _video_opt = types.ModuleType("_video_opt")
    _video_opt._HAS_VIDEO_OPT = False
    _video_opt.default_timebase = Fraction(0, 1)

    def _read_video_stub(filename, start_pts, end_pts, pts_unit):
        raise RuntimeError("video_reader backend is not available. Use backend='pyav'.")

    def _read_video_timestamps_stub(filename, pts_unit):
        raise RuntimeError("video_reader backend is not available. Use backend='pyav'.")

    _video_opt._read_video = _read_video_stub
    _video_opt._read_video_timestamps = _read_video_timestamps_stub

from .image import (
    decode_avif,
    decode_gif,
    decode_heic,
    decode_image,
    decode_jpeg,
    decode_png,
    decode_webp,
    encode_jpeg,
    encode_png,
    ImageReadMode,
    read_file,
    read_image,
    write_file,
    write_jpeg,
    write_png,
)
from .video import read_video, read_video_timestamps, write_video


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
    "_HAS_CPU_VIDEO_DECODER",
    "_HAS_VIDEO_OPT",
    "_read_video_clip_from_memory",
    "_read_video_meta_data",
    "VideoMetaData",
    "Timebase",
    "ImageReadMode",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "decode_avif",
    "decode_heic",
    "decode_webp",
    "decode_gif",
    "encode_jpeg",
    "encode_png",
    "read_file",
    "read_image",
    "write_file",
    "write_jpeg",
    "write_png",
    "Video",
    "VideoReader",
]
