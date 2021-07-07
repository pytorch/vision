import torch

from ._internally_replaced_utils import _get_extension_path


_HAS_OPS = False


def _has_ops():
    return False


try:
    lib_path = _get_extension_path('_C')
    torch.ops.load_library(lib_path)
    _HAS_OPS = True

    def _has_ops():  # noqa: F811
        return True
except (ImportError, OSError):
    pass


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops. This can happen if your PyTorch and "
            "torchvision versions are incompatible, or if you had errors while compiling "
            "torchvision from source. For further information on the compatible versions, check "
            "https://github.com/pytorch/vision#installation for the compatibility matrix. "
            "Please check your PyTorch version with torch.__version__ and your torchvision "
            "version with torchvision.__version__ and verify if they are compatible, and if not "
            "please reinstall torchvision so that it matches your PyTorch install."
        )


def _check_cuda_version():
    """
    Make sure that CUDA versions match between the pytorch install and torchvision install
    """
    if not _HAS_OPS:
        return -1
    import torch
    _version = torch.ops.torchvision._cuda_version()
    if _version != -1 and torch.version.cuda is not None:
        tv_version = str(_version)
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
    return _version


_check_cuda_version()
