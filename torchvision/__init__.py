from torchvision import models
from torchvision import datasets
from torchvision import ops
from torchvision import transforms
from torchvision import utils

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

_image_backend = 'PIL'


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


def _check_cuda_matches():
    """
    Make sure that CUDA versions match between the pytorch install and torchvision install
    """
    import torch
    from torchvision import _C
    if hasattr(_C, "CUDA_VERSION") and torch.version.cuda is not None:
        tv_version = str(_C.CUDA_VERSION)
        if int(tv_version) < 10000:
            tv_major = int(tv_version[0])
            tv_minor = int(tv_version[2])
        else:
            tv_major = int(tv_version[0:2])
            tv_minor = int(tv_version[3])
        t_version = torch.version.cuda
        t_version = t_version.split('.')
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != tv_major or t_minor != tv_minor:
            raise RuntimeError("Detected that PyTorch and torchvision were compiled with different CUDA versions. "
                               "PyTorch has CUDA Version={}.{} and torchvision has CUDA Version={}.{}. "
                               "Please reinstall the torchvision that matches your PyTorch install."
                               .format(t_major, t_minor, tv_major, tv_minor))


_check_cuda_matches()
