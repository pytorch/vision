from ..extension import _load_library


try:
    _load_library("Decoder")
    _HAS_VIDEO_DECODER = True
except (ImportError, OSError):
    _HAS_VIDEO_DECODER = False
