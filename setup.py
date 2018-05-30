#!/usr/bin/env python
import os
import io
import re
import shutil
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import glob

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open('README.rst').read()

VERSION = find_version('torchvision', '__init__.py')

requirements = [
    'numpy',
    'six',
    'torch',
    'tqdm'
]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'torchvision._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros
        )
    ]

    return ext_modules

pillow_ver = ' >= 4.1.1'
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req + pillow_ver)

setup(
    # Metadata
    name='torchvision',
    version=VERSION,
    author='PyTorch Core Team',
    author_email='soumith@pytorch.org',
    url='https://github.com/pytorch/vision',
    description='image and video datasets and models for torch deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,

    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)
