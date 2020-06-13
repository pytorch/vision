import os
import io
import re
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import subprocess
import distutils.command.clean
import distutils.spawn
import glob
import shutil

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from torch.utils.hipify import hipify_python


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


version = '0.7.0a0'
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
    pytorch_dep,
]

pillow_ver = ' >= 4.1.1'
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req + pillow_ver)


def get_extensions():
    try:
        include_dirs = [os.environ['LIBRARY_INC']]
    except KeyError:
        include_dirs = []

    try:
        library_dirs = [os.environ['LIBRARY_LIB']]
    except KeyError:
        library_dirs = []

    # Check if building on Conda
    build_prefix = os.environ.get('BUILD_PREFIX', None)
    is_conda_build = build_prefix is not None
    print('Running build on conda-build: {0}'.format(is_conda_build))
    if is_conda_build:
        # Add LibPNG headers/libraries
        png_include = os.path.join(build_prefix, 'include', 'libpng16')
        print('PNG found? {0}'.format(os.path.isdir(png_include)))
        conda_library = os.path.join(build_prefix, 'lib')
        include_dirs.append(png_include)
        library_dirs.append(conda_library)
    else:
        # Check if using Anaconda to produce wheels
        conda = distutils.spawn.find_executable('conda')
        is_conda = conda is not None
        print('Running build on conda: {0}'.format(is_conda))
        if is_conda:
            python_executable = sys.executable
            if os.name == 'nt':
                env_folder = os.path.dirname(python_executable)
                env_library_path = os.path.join(env_folder, 'Library')
                env_include_folder = os.path.join(env_library_path, 'include')
                env_png_folder = env_include_folder
                png_header = os.path.join(env_include_folder, 'png.h')
                env_lib_folder = os.path.join(env_library_path, 'lib')
                print('PNG found? {0}'.format(os.path.isfile(png_header)))
            else:
                env_bin_folder = os.path.dirname(python_executable)
                env_folder = os.path.dirname(env_bin_folder)
                env_lib_folder = os.path.join(env_folder, 'lib')
                env_png_folder = os.path.join(env_folder, 'include', 'libpng16')
                print('PNG found? {0}'.format(os.path.isdir(env_png_folder)))
            include_dirs.append(env_png_folder)
            library_dirs.append(env_lib_folder)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    image_src = glob.glob(os.path.join(extensions_dir, 'cpu', 'image', '*.cpp'))

    is_rocm_pytorch = False
    if torch.__version__ >= '1.5':
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

    if is_rocm_pytorch:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="torchvision/csrc/cuda/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )
        source_cuda = glob.glob(os.path.join(extensions_dir, 'hip', '*.hip'))
        # Copy over additional files
        shutil.copy("torchvision/csrc/cuda/cuda_helpers.h", "torchvision/csrc/hip/cuda_helpers.h")
        shutil.copy("torchvision/csrc/cuda/vision_cuda.h", "torchvision/csrc/hip/vision_cuda.h")

    else:
        source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu + image_src
    extension = CppExtension

    compile_cpp_tests = os.getenv('WITH_CPP_MODELS_TEST', '0') == '1'
    if compile_cpp_tests:
        test_dir = os.path.join(this_dir, 'test')
        models_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'models')
        test_file = glob.glob(os.path.join(test_dir, '*.cpp'))
        source_models = glob.glob(os.path.join(models_dir, '*.cpp'))

        test_file = [os.path.join(test_dir, s) for s in test_file]
        source_models = [os.path.join(models_dir, s) for s in source_models]
        tests = test_file + source_models
        tests_include_dirs = [test_dir, models_dir]

    define_macros = []

    extra_compile_args = {}
    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) \
            or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda
        if not is_rocm_pytorch:
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            if nvcc_flags == '':
                nvcc_flags = []
            else:
                nvcc_flags = nvcc_flags.split(' ')
        else:
            define_macros += [('WITH_HIP', None)]
            nvcc_flags = []
        extra_compile_args = {
            'cxx': [],
            'nvcc': nvcc_flags,
        }

    if sys.platform == 'win32':
        define_macros += [('torchvision_EXPORTS', None)]

        extra_compile_args.setdefault('cxx', [])
        extra_compile_args['cxx'].append('/MP')

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs += [extensions_dir]

    ext_modules = [
        extension(
            'torchvision._C',
            sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['png'],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
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

    # Image reading extension
    # image_src_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'cpu', 'image')
    # image_src = glob.glob(os.path.join(image_src_dir, "*.cpp"))
    # ext_modules.append(extension(
    #     'torchvision.image',
    #     image_src,
    #     include_dirs=include_dirs + [image_src_dir],
    #     library_dirs=library_dirs,
    #     define_macros=define_macros,
    #     extra_compile_args=extra_compile_args
    # ))

    ffmpeg_exe = distutils.spawn.find_executable('ffmpeg')
    has_ffmpeg = ffmpeg_exe is not None

    if has_ffmpeg:
        ffmpeg_bin = os.path.dirname(ffmpeg_exe)
        ffmpeg_root = os.path.dirname(ffmpeg_bin)
        ffmpeg_include_dir = os.path.join(ffmpeg_root, 'include')

        # TorchVision base decoder + video reader
        video_reader_src_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'cpu', 'video_reader')
        video_reader_src = glob.glob(os.path.join(video_reader_src_dir, "*.cpp"))
        base_decoder_src_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'cpu', 'decoder')
        base_decoder_src = glob.glob(
            os.path.join(base_decoder_src_dir, "*.cpp"))
        # exclude tests
        base_decoder_src = [x for x in base_decoder_src if '_test.cpp' not in x]

        combined_src = video_reader_src + base_decoder_src

        ext_modules.append(
            CppExtension(
                'torchvision.video_reader',
                combined_src,
                include_dirs=[
                    base_decoder_src_dir,
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
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
        'clean': clean,
    }
)
