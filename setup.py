from __future__ import print_function
import os
import io
import re
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import subprocess
import distutils.command.clean
import distutils.spawn
import multiprocessing
import glob
import shutil

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


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


version = '0.6.0a0'
sha = 'Unknown'
package_name = 'torchvision'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'torchvision', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("from torchvision.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


write_version_file()

readme = open('README.rst').read()

pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    'numpy',
    'six',
    pytorch_dep,
]

pillow_ver = ' >= 4.1.1'
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req + pillow_ver)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_image_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', 'image', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu

    libraries = []
    extra_compile_args = {}
    third_party_search_directories = []
    runtime_library_dirs = []

    if sys.platform.startswith('linux'):
        sources = sources + source_image_cpu
        libraries.append('turbojpeg')
        third_party_search_directories.append(os.path.join(cwd, "third_party/libjpeg-turbo"))
        runtime_library_dirs = ['.']

    extension = CppExtension

    compile_cpp_tests = os.getenv('WITH_CPP_MODELS_TEST', '0') == '1'
    if compile_cpp_tests:
        test_dir = os.path.join(this_dir, 'test')
        models_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'models')
        test_file = glob.glob(os.path.join(test_dir, '*.cpp'))
        source_models = glob.glob(os.path.join(models_dir, '*.cpp'))
        extra_compile_args.setdefault('cxx', [])

        test_file = [os.path.join(test_dir, s) for s in test_file]
        source_models = [os.path.join(models_dir, s) for s in source_models]
        tests = test_file + source_models
        tests_include_dirs = [test_dir, models_dir]

    define_macros = []

    extra_compile_args = {}
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args = {
            'cxx': [],
            'nvcc': nvcc_flags,
        }

    if sys.platform == 'win32':
        define_macros += [('torchvision_EXPORTS', None)]

        extra_compile_args.setdefault('cxx', [])
        extra_compile_args['cxx'].append('/MP')

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ffmpeg_exe = distutils.spawn.find_executable('ffmpeg')
    has_ffmpeg = ffmpeg_exe is not None
    if has_ffmpeg:
        ffmpeg_bin = os.path.dirname(ffmpeg_exe)
        ffmpeg_root = os.path.dirname(ffmpeg_bin)
        ffmpeg_include_dir = os.path.join(ffmpeg_root, 'include')

        # TorchVision video reader
        video_reader_src_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'cpu', 'video_reader')
        video_reader_src = glob.glob(os.path.join(video_reader_src_dir, "*.cpp"))

    ext_modules = [
        extension(
            'torchvision._C',
            sources,
            libraries=libraries,
            library_dirs=third_party_search_directories,
            include_dirs=include_dirs + third_party_search_directories,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            runtime_library_dirs=runtime_library_dirs
        )
    ]
    if compile_cpp_tests:
        ext_modules.append(
            extension(
                'torchvision._C_tests',
                tests,
                include_dirs=tests_include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        )
    if has_ffmpeg:
        ext_modules.append(
            CppExtension(
                'torchvision.video_reader',
                video_reader_src,
                include_dirs=[
                    video_reader_src_dir,
                    ffmpeg_include_dir,
                    extensions_dir,
                ],
                libraries=[
                    'avcodec',
                    'avformat',
                    'avutil',
                    'swresample',
                    'swscale',
                ],
                extra_compile_args=["-std=c++14"],
                extra_link_args=["-std=c++14"],
            )
        )

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


def throw_of_failure(command):
    ret = os.system(command)
    if ret != 0:
        raise Exception("{} failed".format(command))


def build_deps():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform.startswith('linux'):
        cpu_count = multiprocessing.cpu_count()
        os.chdir("third_party/libjpeg-turbo/")
        throw_of_failure('cmake .')
        throw_of_failure("cmake --build . -- -j {}".format(cpu_count))
        os.chdir(this_dir)


def build_ext_with_dependencies(self):
    build_deps()
    return BuildExtension.with_options(no_python_abi_suffix=True)(self)


data_files = []
if sys.platform.startswith('linux'):
    data_files = [
        ('torchvision', [
            'third_party/libjpeg-turbo/libturbojpeg.so.0',
            'third_party/libjpeg-turbo/libturbojpeg.so'])
    ]

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

    zip_safe=False,
    install_requires=requirements,
    extras_require={
        "scipy": ["scipy"],
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': build_ext_with_dependencies,
        'clean': clean,
    },
    data_files=data_files)
