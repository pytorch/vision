from modulefinder import Module

import torch

# Don't re-order these, we need to load the _C extension (done when importing
# .extension) before entering _meta_registrations.
from . import extension  # usort:skip  # noqa: F401
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


_image_backend = "PIL"

_video_backend = "pyav"


def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ["PIL", "accimage"]:
        raise ValueError(f"Invalid backend '{backend}'. Options are 'PIL' and 'accimage'")
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend


def set_video_backend(backend):
    """
    Specifies the package used to decode videos.

    Args:
        backend (string): Name of the video backend. Only 'pyav' is supported.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
    """
    pass


def get_video_backend():
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend. Currently only 'pyav' is supported.
    """

    return _video_backend


def _is_tracing():
    return torch._C._get_tracing_state()


def disable_beta_transforms_warning():
    # Noop, only exists to avoid breaking existing code.
    # See https://github.com/pytorch/vision/issues/7896
    pass
