from __future__ import print_function
import os
import io
import re
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import subprocess
import distutils.command.clean
import glob
import shutil

import torch
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


version = '0.3.0'
sha = 'Unknown'
package_name = os.getenv('TORCHVISION_PACKAGE_NAME', 'torchvision')

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('TORCHVISION_BUILD_VERSION'):
    assert os.getenv('TORCHVISION_BUILD_NUMBER') is not None
    build_number = int(os.getenv('TORCHVISION_BUILD_NUMBER'))
    version = os.getenv('TORCHVISION_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'torchvision', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("from torchvision import _C\n")
        f.write("if hasattr(_C, 'CUDA_VERSION'):\n")
        f.write("    cuda = _C.CUDA_VERSION\n")


write_version_file()

readme = open('README.rst').read()

pytorch_package_name = os.getenv('TORCHVISION_PYTORCH_DEPENDENCY_NAME', 'torch >= 1.1.0')

requirements = [
    'numpy',
    'six',
    pytorch_package_name,
]

pillow_ver = ' >= 4.1.1'
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req + pillow_ver)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    extra_compile_args = {}
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args = {
            'cxx': ['-O0'],
            'nvcc': nvcc_flags,
        }

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'torchvision._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


setup(
    # Metadata
    name=package_name,
    version=version,
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
    extras_require={
        "scipy": ["scipy"],
    },
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension, 'clean': clean}
)
