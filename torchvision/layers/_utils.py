import os.path

import torch

try:
    from torch.utils.cpp_extension import load as load_ext
except ImportError:
    raise ImportError("The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions(source_cpu, source_gpu=None):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.dirname(this_dir)
    this_dir = os.path.join(this_dir, 'csrc')
    source = source_cpu
    extra_cflags = []
    if torch.cuda.is_available() and source_gpu is not None:
        source.extend(source_gpu)
        extra_cflags = ['-DWITH_CUDA']
    source = [os.path.join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext('torchvision', source, extra_cflags=extra_cflags, extra_include_paths=extra_include_paths)

main_file = ['vision.cpp']
source_cpu = ['cpu/nms_cpu.cpp', 'cpu/ROIAlign_cpu.cpp']
source_cuda = ['cuda/ROIAlign_cuda.cu', 'cuda/ROIPool_cuda.cu']

_C = _load_C_extensions(source_cpu + main_file, source_cuda)

