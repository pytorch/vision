import distutils.command.clean
import distutils.spawn
import glob
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension, ROCM_HOME

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
FORCE_MPS = os.getenv("FORCE_MPS", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
USE_PNG = os.getenv("TORCHVISION_USE_PNG", "1") == "1"
USE_JPEG = os.getenv("TORCHVISION_USE_JPEG", "1") == "1"
USE_NVJPEG = os.getenv("TORCHVISION_USE_NVJPEG", "1") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", None)
USE_FFMPEG = os.getenv("TORCHVISION_USE_FFMPEG", "1") == "1"
USE_VIDEO_CODEC = os.getenv("TORCHVISION_USE_VIDEO_CODEC", "1") == "1"

TORCHVISION_INCLUDE = os.environ.get("TORCHVISION_INCLUDE", "")
TORCHVISION_LIBRARY = os.environ.get("TORCHVISION_LIBRARY", "")
TORCHVISION_INCLUDE = TORCHVISION_INCLUDE.split(os.pathsep) if TORCHVISION_INCLUDE else []
TORCHVISION_LIBRARY = TORCHVISION_LIBRARY.split(os.pathsep) if TORCHVISION_LIBRARY else []

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "torchvision/csrc"
IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA

PACKAGE_NAME = "torchvision"

print("Torchvision build configuration:")
print(f"{FORCE_CUDA = }")
print(f"{FORCE_MPS = }")
print(f"{DEBUG = }")
print(f"{USE_PNG = }")
print(f"{USE_JPEG = }")
print(f"{USE_NVJPEG = }")
print(f"{NVCC_FLAGS = }")
print(f"{USE_FFMPEG = }")
print(f"{USE_VIDEO_CODEC = }")
print(f"{TORCHVISION_INCLUDE = }")
print(f"{TORCHVISION_LIBRARY = }")
print(f"{IS_ROCM = }")
print(f"{BUILD_CUDA_SOURCES = }")


def get_version():
    with open(ROOT_DIR / "version.txt") as f:
        version = f.readline().strip()
    sha = "Unknown"

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def write_version_file(version, sha):
    # Exists for BC, probably completely useless.
    with open(ROOT_DIR / "torchvision/version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write("from torchvision.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


def get_requirements():
    def get_dist(pkgname):
        try:
            return get_distribution(pkgname)
        except DistributionNotFound:
            return None

    pytorch_dep = "torch"
    if os.getenv("PYTORCH_VERSION"):
        pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")

    requirements = [
        # TODO: Remove <2 constraint! https://github.com/pytorch/vision/issues/8531
        "numpy<2" if sys.platform == "win32" else "numpy",
        pytorch_dep,
    ]

    # Excluding 8.3.* because of https://github.com/pytorch/vision/issues/4934
    pillow_ver = " >= 5.3.0, !=8.3.*"
    pillow_req = "pillow-simd" if get_dist("pillow-simd") is not None else "pillow"
    requirements.append(pillow_req + pillow_ver)

    return requirements


def get_macros_and_flags():
    define_macros = []
    extra_compile_args = {"cxx": []}
    if BUILD_CUDA_SOURCES:
        if IS_ROCM:
            define_macros += [("WITH_HIP", None)]
            nvcc_flags = []
        else:
            define_macros += [("WITH_CUDA", None)]
            if NVCC_FLAGS is None:
                nvcc_flags = []
            else:
                nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("torchvision_EXPORTS", None)]
        extra_compile_args["cxx"].append("/MP")

    if DEBUG:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [f for f in nvcc_flags if not ("-O" in f or "-g" in f)]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")
    else:
        extra_compile_args["cxx"].append("-g0")

    return define_macros, extra_compile_args


def make_C_extension():

    sources = (
        list(CSRS_DIR.glob("*.cpp"))
        + list(CSRS_DIR.glob("ops/*.cpp"))
        + list(CSRS_DIR.glob("ops/autocast/*.cpp"))
        + list(CSRS_DIR.glob("ops/autograd/*.cpp"))
        + list(CSRS_DIR.glob("ops/cpu/*.cpp"))
        + list(CSRS_DIR.glob("ops/quantized/cpu/*.cpp"))
    )
    mps_sources = list(CSRS_DIR.glob("ops/mps/*.mm"))

    if IS_ROCM:
        from torch.utils.hipify import hipify_python

        hipify_python.hipify(
            project_directory=str(ROOT_DIR),
            output_directory=str(ROOT_DIR),
            includes="torchvision/csrc/ops/cuda/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )
        cuda_sources = list(CSRS_DIR.glob("ops/hip/*.hip"))
        for header in CSRS_DIR.glob("ops/cuda/*.h"):
            shutil.copy(str(header), str(CSRS_DIR / "ops/hip"))
    else:
        cuda_sources = list(CSRS_DIR.glob("ops/cuda/*.cu"))

    if BUILD_CUDA_SOURCES:
        Extension = CUDAExtension
        sources += cuda_sources
    else:
        Extension = CppExtension
        if torch.backends.mps.is_available() or FORCE_MPS:
            sources += mps_sources

    define_macros, extra_compile_args = get_macros_and_flags()
    return Extension(
        name="torchvision._C",
        sources=sorted(str(s) for s in sources),
        include_dirs=[CSRS_DIR],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def find_libpng():
    # Returns (found, include dir, library dir, library name)
    if sys.platform in ("linux", "darwin"):
        libpng_config = shutil.which("libpng-config")
        if libpng_config is None:
            warnings.warn("libpng-config not found")
            return False, None, None, None
        min_version = parse_version("1.6.0")
        png_version = parse_version(
            subprocess.run([libpng_config, "--version"], stdout=subprocess.PIPE).stdout.strip().decode("utf-8")
        )
        if png_version < min_version:
            warnings.warn("libpng version {png_version} is less than minimum required version {min_version}")
            return False, None, None, None

        include_dir = (
            subprocess.run([libpng_config, "--I_opts"], stdout=subprocess.PIPE)
            .stdout.strip()
            .decode("utf-8")
            .split("-I")[1]
        )
        library_dir = subprocess.run([libpng_config, "--libdir"], stdout=subprocess.PIPE).stdout.strip().decode("utf-8")
        library = "png"
    else:  # Windows
        pngfix = shutil.which("pngfix")
        if pngfix is None:
            warnings.warn("pngfix not found")
            return False, None, None, None
        pngfix_dir = Path(pngfix).absolute().parent.parent

        library_dir = str(pngfix_dir / "lib")
        include_dir = str(pngfix_dir / "include/libpng16")
        library = "libpng"

    return True, include_dir, library_dir, library


def find_libjpeg():
    # returns (found, include dir, library dir)
    # if include dir or library dir is None, it means that the library is in
    # standard paths and don't need to be added to compiler / linker search
    # paths

    library_header = "jpeglib.h"
    searching_for = f"Searching for {library_header}"

    for folder in TORCHVISION_INCLUDE:
        if (Path(folder) / library_header).exists():
            print(f"{searching_for}. Found in TORCHVISION_INCLUDE.")
            return True, None, None
    print(f"{searching_for}. Didn't find in TORCHVISION_INCLUDE.")

    # Try conda-related prefixes. If BUILD_PREFIX is set it means conda-build is
    # being run. If CONDA_PREFIX is set then we're in a conda environment.
    for prefix_env_var in ("BUILD_PREFIX", "CONDA_PREFIX"):
        if (prefix := os.environ.get(prefix_env_var)) is not None:
            prefix = Path(prefix)
            if sys.platform == "win32":
                prefix = prefix / "Library"
            include_dir = prefix / "include"
            library_dir = prefix / "lib"
            if (include_dir / library_header).exists():
                print(f"{searching_for}. Found in {prefix_env_var}.")
                return True, str(include_dir), str(library_dir)
        print(f"{searching_for}. Didn't find in {prefix_env_var}.")

    if sys.platform == "linux":
        prefixes = ("/usr/include", "/usr/local/include")
        if any((Path(prefix) / library_header).exists() for prefix in prefixes):
            print(f"{searching_for}. Found in {prefixes}.")
            return True, None, None
        print(f"{searching_for}. Didn't find in {prefixes}.")

    return False, None, None


def make_image_extension():

    include_dirs = TORCHVISION_INCLUDE.copy()
    library_dirs = TORCHVISION_LIBRARY.copy()

    libraries = []
    define_macros, extra_compile_args = get_macros_and_flags()

    image_dir = CSRS_DIR / "io/image"
    sources = list(image_dir.glob("*.cpp")) + list(image_dir.glob("cpu/*.cpp")) + list(image_dir.glob("cpu/giflib/*.c"))

    if IS_ROCM:
        sources += list(image_dir.glob("hip/*.cpp"))
        # we need to exclude this in favor of the hipified source
        sources.remove(image_dir / "image.cpp")
    else:
        sources += list(image_dir.glob("cuda/*.cpp"))

    Extension = CppExtension

    if USE_PNG:
        png_found, png_include_dir, png_library_dir, png_library = find_libpng()
        if png_found:
            print("Building torchvision with PNG support")
            print(f"{png_include_dir = }")
            print(f"{png_library_dir = }")
            include_dirs.append(png_include_dir)
            library_dirs.append(png_library_dir)
            libraries.append(png_library)
            define_macros += [("PNG_FOUND", 1)]
        else:
            warnings.warn("Building torchvision without PNG support")

    if USE_JPEG:
        jpeg_found, jpeg_include_dir, jpeg_library_dir = find_libjpeg()
        if jpeg_found:
            print("Building torchvision with JPEG support")
            print(f"{jpeg_include_dir = }")
            print(f"{jpeg_library_dir = }")
            if jpeg_include_dir is not None and jpeg_library_dir is not None:
                # if those are None it means they come from standard paths that are already in the search paths, which we don't need to re-add.
                include_dirs.append(jpeg_include_dir)
                library_dirs.append(jpeg_library_dir)
            libraries.append("jpeg")
            define_macros += [("JPEG_FOUND", 1)]
        else:
            warnings.warn("Building torchvision without JPEG support")

    if USE_NVJPEG and torch.cuda.is_available():
        nvjpeg_found = CUDA_HOME is not None and (Path(CUDA_HOME) / "include/nvjpeg.h").exists()

        if nvjpeg_found:
            print("Building torchvision with NVJPEG image support")
            libraries.append("nvjpeg")
            define_macros += [("NVJPEG_FOUND", 1)]
            Extension = CUDAExtension
        else:
            warnings.warn("Building torchvision without NVJPEG support")

    return Extension(
        name="torchvision.image",
        sources=sorted(str(s) for s in sources),
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )


def make_video_decoders_extensions():
    # Locating ffmpeg
    ffmpeg_exe = shutil.which("ffmpeg")
    has_ffmpeg = ffmpeg_exe is not None
    ffmpeg_version = None
    # FIXME: Building torchvision with ffmpeg on MacOS or with Python 3.9
    # FIXME: causes crash. See the following GitHub issues for more details.
    # FIXME: https://github.com/pytorch/pytorch/issues/65000
    # FIXME: https://github.com/pytorch/vision/issues/3367
    if sys.platform != "linux" or (sys.version_info.major == 3 and sys.version_info.minor == 9):
        has_ffmpeg = False
    if has_ffmpeg:
        try:
            # This is to check if ffmpeg is installed properly.
            ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"])
        except subprocess.CalledProcessError:
            print("Building torchvision without ffmpeg support")
            print("  Error fetching ffmpeg version, ignoring ffmpeg.")
            has_ffmpeg = False

    use_ffmpeg = USE_FFMPEG and has_ffmpeg

    if use_ffmpeg:
        ffmpeg_libraries = {"libavcodec", "libavformat", "libavutil", "libswresample", "libswscale"}

        ffmpeg_bin = os.path.dirname(ffmpeg_exe)
        ffmpeg_root = os.path.dirname(ffmpeg_bin)
        ffmpeg_include_dir = os.path.join(ffmpeg_root, "include")
        ffmpeg_library_dir = os.path.join(ffmpeg_root, "lib")

        gcc = os.environ.get("CC", shutil.which("gcc"))
        platform_tag = subprocess.run([gcc, "-print-multiarch"], stdout=subprocess.PIPE)
        platform_tag = platform_tag.stdout.strip().decode("utf-8")

        if platform_tag:
            # Most probably a Debian-based distribution
            ffmpeg_include_dir = [ffmpeg_include_dir, os.path.join(ffmpeg_include_dir, platform_tag)]
            ffmpeg_library_dir = [ffmpeg_library_dir, os.path.join(ffmpeg_library_dir, platform_tag)]
        else:
            ffmpeg_include_dir = [ffmpeg_include_dir]
            ffmpeg_library_dir = [ffmpeg_library_dir]

        for library in ffmpeg_libraries:
            library_found = False
            for search_path in ffmpeg_include_dir + TORCHVISION_INCLUDE:
                full_path = os.path.join(search_path, library, "*.h")
                library_found |= len(glob.glob(full_path)) > 0

            if not library_found:
                print("Building torchvision without ffmpeg support")
                print(f"  {library} header files were not found, disabling ffmpeg support")
                use_ffmpeg = False
    else:
        print("Building torchvision without ffmpeg support")

    extensions = []
    if use_ffmpeg:
        print("Building torchvision with ffmpeg support")
        print(f"  ffmpeg version: {ffmpeg_version}")
        print(f"  ffmpeg include path: {ffmpeg_include_dir}")
        print(f"  ffmpeg library_dir: {ffmpeg_library_dir}")

        # TorchVision base decoder + video reader
        video_reader_src_dir = os.path.join(ROOT_DIR, "torchvision", "csrc", "io", "video_reader")
        video_reader_src = glob.glob(os.path.join(video_reader_src_dir, "*.cpp"))
        base_decoder_src_dir = os.path.join(ROOT_DIR, "torchvision", "csrc", "io", "decoder")
        base_decoder_src = glob.glob(os.path.join(base_decoder_src_dir, "*.cpp"))
        # Torchvision video API
        videoapi_src_dir = os.path.join(ROOT_DIR, "torchvision", "csrc", "io", "video")
        videoapi_src = glob.glob(os.path.join(videoapi_src_dir, "*.cpp"))
        # exclude tests
        base_decoder_src = [x for x in base_decoder_src if "_test.cpp" not in x]

        combined_src = video_reader_src + base_decoder_src + videoapi_src

        extensions.append(
            CppExtension(
                "torchvision.video_reader",
                combined_src,
                include_dirs=[
                    base_decoder_src_dir,
                    video_reader_src_dir,
                    videoapi_src_dir,
                    str(CSRS_DIR),
                    *ffmpeg_include_dir,
                    *TORCHVISION_INCLUDE,
                ],
                library_dirs=ffmpeg_library_dir + TORCHVISION_LIBRARY,
                libraries=[
                    "avcodec",
                    "avformat",
                    "avutil",
                    "swresample",
                    "swscale",
                ],
                extra_compile_args=["-std=c++17"] if os.name != "nt" else ["/std:c++17", "/MP"],
                extra_link_args=["-std=c++17" if os.name != "nt" else "/std:c++17"],
            )
        )

    # Locating video codec
    # CUDA_HOME should be set to the cuda root directory.
    # TORCHVISION_INCLUDE and TORCHVISION_LIBRARY should include the location to
    # video codec header files and libraries respectively.
    video_codec_found = (
        BUILD_CUDA_SOURCES
        and CUDA_HOME is not None
        and any([os.path.exists(os.path.join(folder, "cuviddec.h")) for folder in TORCHVISION_INCLUDE])
        and any([os.path.exists(os.path.join(folder, "nvcuvid.h")) for folder in TORCHVISION_INCLUDE])
        and any([os.path.exists(os.path.join(folder, "libnvcuvid.so")) for folder in TORCHVISION_LIBRARY])
    )

    use_video_codec = USE_VIDEO_CODEC and video_codec_found
    if (
        use_video_codec
        and use_ffmpeg
        and any([os.path.exists(os.path.join(folder, "libavcodec", "bsf.h")) for folder in ffmpeg_include_dir])
    ):
        print("Building torchvision with video codec support")
        gpu_decoder_path = os.path.join(CSRS_DIR, "io", "decoder", "gpu")
        gpu_decoder_src = glob.glob(os.path.join(gpu_decoder_path, "*.cpp"))
        cuda_libs = os.path.join(CUDA_HOME, "lib64")
        cuda_inc = os.path.join(CUDA_HOME, "include")

        _, extra_compile_args = get_macros_and_flags()
        extensions.append(
            CUDAExtension(
                "torchvision.Decoder",
                gpu_decoder_src,
                include_dirs=[CSRS_DIR] + TORCHVISION_INCLUDE + [gpu_decoder_path] + [cuda_inc] + ffmpeg_include_dir,
                library_dirs=ffmpeg_library_dir + TORCHVISION_LIBRARY + [cuda_libs],
                libraries=[
                    "avcodec",
                    "avformat",
                    "avutil",
                    "swresample",
                    "swscale",
                    "nvcuvid",
                    "cuda",
                    "cudart",
                    "z",
                    "pthread",
                    "dl",
                    "nppicc",
                ],
                extra_compile_args=extra_compile_args,
            )
        )
    else:
        print("Building torchvision without video codec support")
        if (
            use_video_codec
            and use_ffmpeg
            and not any([os.path.exists(os.path.join(folder, "libavcodec", "bsf.h")) for folder in ffmpeg_include_dir])
        ):
            print(
                "  The installed version of ffmpeg is missing the header file 'bsf.h' which is "
                "  required for GPU video decoding. Please install the latest ffmpeg from conda-forge channel:"
                "   `conda install -c conda-forge ffmpeg`."
            )

    return extensions


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    version, sha = get_version()
    write_version_file(version, sha)

    print(f"Building wheel {PACKAGE_NAME}-{version}")

    with open("README.md") as f:
        readme = f.read()

    extensions = [
        make_C_extension(),
        make_image_extension(),
        *make_video_decoders_extensions(),
    ]

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="PyTorch Core Team",
        author_email="soumith@pytorch.org",
        url="https://github.com/pytorch/vision",
        description="image and video datasets and models for torch deep learning",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD",
        packages=find_packages(exclude=("test",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.dylib", "*.so", "prototype/datasets/_builtin/*.categories"]},
        zip_safe=False,
        install_requires=get_requirements(),
        extras_require={
            "gdown": ["gdown>=4.7.3"],
            "scipy": ["scipy"],
        },
        ext_modules=extensions,
        python_requires=">=3.8",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
