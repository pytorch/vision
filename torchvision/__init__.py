import os
import warnings
from modulefinder import Module

import torch

# Don't re-order these, we need to load the _C extension (done when importing
# .extensions) before entering _meta_registrations.
from .extension import _HAS_OPS  # usort:skip
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip

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


def _is_tracing():
    return torch._C._get_tracing_state()


def disable_beta_transforms_warning():
    # Noop, only exists to avoid breaking existing code.
    # See https://github.com/pytorch/vision/issues/7896
    pass
