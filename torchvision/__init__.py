import os
import warnings
from modulefinder import Module

import torch
from torchvision import datasets, io, models, ops, transforms, utils

from .extension import _HAS_OPS

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


# Check if torchvision is being imported within the root folder
if not _HAS_OPS and os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "torchvision"
):
    message = (
        "You are importing torchvision within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "torchvision project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))

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
        backend (string): Name of the video backend. one of {'pyav', 'video_reader'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It generally decodes faster than :mod:`pyav`, but is perhaps less robust.

    .. note::
        Building with FFMPEG is disabled by default in the latest `main`. If you want to use the 'video_reader'
        backend, please compile torchvision from source.
    """
    global _video_backend
    if backend not in ["pyav", "video_reader", "cuda"]:
        raise ValueError("Invalid video backend '%s'. Options are 'pyav', 'video_reader' and 'cuda'" % backend)
    if backend == "video_reader" and not io._HAS_VIDEO_OPT:
        # TODO: better messages
        message = "video_reader video backend is not available. Please compile torchvision from source and try again"
        raise RuntimeError(message)
    elif backend == "cuda" and not io._HAS_GPU_VIDEO_DECODER:
        # TODO: better messages
        message = "cuda video backend is not available."
        raise RuntimeError(message)
    else:
        _video_backend = backend


def get_video_backend():
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend. one of {'pyav', 'video_reader'}.
    """

    return _video_backend


def _is_tracing():
    return torch._C._get_tracing_state()


_WARN_ABOUT_BETA_TRANSFORMS = True
_BETA_TRANSFORMS_WARNING = (
    "The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. "
    "While we do not expect major breaking changes, some APIs may still change "
    "according to user feedback. Please submit any feedback you may have in "
    "this issue: https://github.com/pytorch/vision/issues/6753, and you can also "
    "check out https://github.com/pytorch/vision/issues/7319 to learn more about "
    "the APIs that we suspect might involve future changes. "
    "You can silence this warning by calling torchvision.disable_beta_transforms_warning()."
)


def disable_beta_transforms_warning():
    global _WARN_ABOUT_BETA_TRANSFORMS
    _WARN_ABOUT_BETA_TRANSFORMS = False
