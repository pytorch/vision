_HAS_OPS = False


def _register_extensions():
    import os
    import imp
    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    _, path, _ = imp.find_module("_C", [lib_dir])
    torch.ops.load_library(path)


try:
    _register_extensions()
    _HAS_OPS = True
except (ImportError, OSError):
    pass


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
