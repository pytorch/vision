import os.path
import glob

import torch

try:
    from torch.utils.cpp_extension import load as load_ext
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError("The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.dirname(this_dir)
    this_dir = os.path.join(this_dir, 'csrc')

    main_file = glob.glob(os.path.join(this_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(this_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(this_dir, 'cuda', '*.cu'))

    source = main_file + source_cpu

    extra_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags = ['-DWITH_CUDA']
    source = [os.path.join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext('torchvision', source, extra_cflags=extra_cflags, extra_include_paths=extra_include_paths)

_C = _load_C_extensions()
