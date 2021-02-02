import warnings
import os

from .extension import _HAS_OPS

from torchvision import models
from torchvision import datasets
from torchvision import ops
from torchvision import transforms
from torchvision import utils
from torchvision import io

import torch

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

# Check if torchvision is being imported within the root folder
if (not _HAS_OPS and os.path.dirname(os.path.realpath(__file__)) ==
        os.path.join(os.path.realpath(os.getcwd()), 'torchvision')):
    message = ('You are importing torchvision within its own root folder ({}). '
               'This is not expected to work and may give errors. Please exit the '
               'torchvision project source and relaunch your python interpreter.')
    warnings.warn(message.format(os.getcwd()))

_image_backend = 'PIL'

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
    if backend not in ['PIL', 'accimage']:
        raise ValueError("Invalid backend '{}'. Options are 'PIL' and 'accimage'"
                         .format(backend))
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
        backend (string): Name of the video backend. one of {'pyav', 'video_reader'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It is generally decoding faster than :mod:`pyav`, but perhaps is less robust.

    .. note::
        Building with FFMPEG is disabled by default in the latest master. If you want to use the 'video_reader'
        backend, please compile torchvision from source.
    """
    global _video_backend
    if backend not in ["pyav", "video_reader"]:
        raise ValueError(
            "Invalid video backend '%s'. Options are 'pyav' and 'video_reader'" % backend
        )
    if backend == "video_reader" and not io._HAS_VIDEO_OPT:
        message = (
            "video_reader video backend is not available."
            " Please compile torchvision from source and try again"
        )
        warnings.warn(message)
    else:
        _video_backend = backend


def get_video_backend():
    return _video_backend


def _is_tracing():
    return torch._C._get_tracing_state()
