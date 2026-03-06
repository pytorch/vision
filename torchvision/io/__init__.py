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
    pass

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


__all__ = [
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
]
