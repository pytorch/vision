import distutils.command.clean
import distutils.spawn
import glob
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
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
USE_WEBP = os.getenv("TORCHVISION_USE_WEBP", "1") == "1"
USE_NVJPEG = os.getenv("TORCHVISION_USE_NVJPEG", "1") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", None)
# Note: the GPU video decoding stuff used to be called "video codec", which
# isn't an accurate or descriptive name considering there are at least 2 other
# video decoding backends in torchvision. I'm renaming this to "gpu video
# decoder" where possible, keeping user facing names (like the env var below) to
# the old scheme for BC.
USE_GPU_VIDEO_DECODER = os.getenv("TORCHVISION_USE_VIDEO_CODEC", "0") == "1"
# Same here: "use ffmpeg" was used to denote "use cpu video decoder".
USE_CPU_VIDEO_DECODER = os.getenv("TORCHVISION_USE_FFMPEG", "0") == "1"

TORCHVISION_INCLUDE = os.environ.get("TORCHVISION_INCLUDE", "")
TORCHVISION_LIBRARY = os.environ.get("TORCHVISION_LIBRARY", "")
TORCHVISION_INCLUDE = TORCHVISION_INCLUDE.split(os.pathsep) if TORCHVISION_INCLUDE else []
TORCHVISION_LIBRARY = TORCHVISION_LIBRARY.split(os.pathsep) if TORCHVISION_LIBRARY else []

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "torchvision/csrc"
IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA

package_name = os.getenv("TORCHVISION_PACKAGE_NAME", "torchvision")

print("Torchvision build configuration:")
print(f"{FORCE_CUDA = }")
print(f"{FORCE_MPS = }")
print(f"{DEBUG = }")
print(f"{USE_PNG = }")
print(f"{USE_JPEG = }")
print(f"{USE_WEBP = }")
print(f"{USE_NVJPEG = }")
print(f"{NVCC_FLAGS = }")
print(f"{USE_CPU_VIDEO_DECODER = }")
print(f"{USE_GPU_VIDEO_DECODER = }")
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
    with open(ROOT_DIR / "torchvision" / "version.py", "w") as f:
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

    pytorch_dep = os.getenv("TORCH_PACKAGE_NAME", "torch")
    if version_pin := os.getenv("PYTORCH_VERSION"):
        pytorch_dep += "==" + version_pin
    elif (version_pin_ge := os.getenv("PYTORCH_VERSION_GE")) and (version_pin_lt := os.getenv("PYTORCH_VERSION_LT")):
        # This branch and the associated env vars exist to help third-party
        # builds like in https://github.com/pytorch/vision/pull/8936. This is
        # supported on a best-effort basis, we don't guarantee that this won't
        # eventually break (and we don't test it.)
        pytorch_dep += f">={version_pin_ge},<{version_pin_lt}"

    requirements = [
        "numpy",
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
                nvcc_flags = shlex.split(NVCC_FLAGS)
        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("torchvision_EXPORTS", None)]
        extra_compile_args["cxx"].append("/MP")
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            extra_compile_args["cxx"].append("-DPy_GIL_DISABLED")

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
    print("Building _C extension")

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
    if sys.platform in ("linux", "darwin", "aix"):
        libpng_config = shutil.which("libpng-config")
        if libpng_config is None:
            warnings.warn("libpng-config not found")
            return False, None, None, None
        min_version = parse_version("1.6.0")
        png_version = parse_version(
            subprocess.run([libpng_config, "--version"], stdout=subprocess.PIPE).stdout.strip().decode("utf-8")
        )
        if png_version < min_version:
            warnings.warn(f"libpng version {png_version} is less than minimum required version {min_version}")
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


def find_library(header):
    # returns (found, include dir, library dir)
    # if include dir or library dir is None, it means that the library is in
    # standard paths and don't need to be added to compiler / linker search
    # paths

    searching_for = f"Searching for {header}"

    for folder in TORCHVISION_INCLUDE:
        if (Path(folder) / header).exists():
            print(f"{searching_for} in {Path(folder) / header}. Found in TORCHVISION_INCLUDE.")
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
            if (include_dir / header).exists():
                print(f"{searching_for}. Found in {prefix_env_var}.")
                return True, str(include_dir), str(library_dir)
        print(f"{searching_for}. Didn't find in {prefix_env_var}.")

    if sys.platform == "linux":
        for prefix in (Path("/usr/include"), Path("/usr/local/include")):
            if (prefix / header).exists():
                print(f"{searching_for}. Found in {prefix}.")
                return True, None, None
            print(f"{searching_for}. Didn't find in {prefix}")

    if sys.platform == "darwin":
        HOMEBREW_PATH = Path("/opt/homebrew")
        include_dir = HOMEBREW_PATH / "include"
        library_dir = HOMEBREW_PATH / "lib"
        if (include_dir / header).exists():
            print(f"{searching_for}. Found in {include_dir}.")
            return True, str(include_dir), str(library_dir)

    return False, None, None


def make_image_extension():
    print("Building image extension")

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
        jpeg_found, jpeg_include_dir, jpeg_library_dir = find_library(header="jpeglib.h")
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

    if USE_WEBP:
        webp_found, webp_include_dir, webp_library_dir = find_library(header="webp/decode.h")
        if webp_found:
            print("Building torchvision with WEBP support")
            print(f"{webp_include_dir = }")
            print(f"{webp_library_dir = }")
            if webp_include_dir is not None and webp_library_dir is not None:
                # if those are None it means they come from standard paths that are already in the search paths, which we don't need to re-add.
                include_dirs.append(webp_include_dir)
                library_dirs.append(webp_library_dir)
            webp_library = "libwebp" if sys.platform == "win32" else "webp"
            libraries.append(webp_library)
            define_macros += [("WEBP_FOUND", 1)]
        else:
            warnings.warn("Building torchvision without WEBP support")

    if USE_NVJPEG and (torch.cuda.is_available() or FORCE_CUDA):
        nvjpeg_found = CUDA_HOME is not None and (Path(CUDA_HOME) / "include/nvjpeg.h").exists()

        if nvjpeg_found:
            print("Building torchvision with NVJPEG image support")
            libraries.append("nvjpeg")
            define_macros += [("NVJPEG_FOUND", 1)]
            Extension = CUDAExtension
        else:
            warnings.warn("Building torchvision without NVJPEG support")
    elif USE_NVJPEG:
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
    print("Building video decoder extensions")

    build_without_extensions_msg = "Building without video decoders extensions."
    if sys.platform != "linux" or (sys.version_info.major == 3 and sys.version_info.minor == 9):
        # FIXME: Building torchvision with ffmpeg on MacOS or with Python 3.9
        # FIXME: causes crash. See the following GitHub issues for more details.
        # FIXME: https://github.com/pytorch/pytorch/issues/65000
        # FIXME: https://github.com/pytorch/vision/issues/3367
        print("Can only build video decoder extensions on linux and Python != 3.9")
        return []

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        print(f"{build_without_extensions_msg} Couldn't find ffmpeg binary.")
        return []

    def find_ffmpeg_libraries():
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
                print(f"{build_without_extensions_msg}")
                print(f"{library} header files were not found.")
                return None, None

        return ffmpeg_include_dir, ffmpeg_library_dir

    ffmpeg_include_dir, ffmpeg_library_dir = find_ffmpeg_libraries()
    if ffmpeg_include_dir is None or ffmpeg_library_dir is None:
        return []

    print("Found ffmpeg:")
    print(f"  ffmpeg include path: {ffmpeg_include_dir}")
    print(f"  ffmpeg library_dir: {ffmpeg_library_dir}")

    extensions = []
    if USE_CPU_VIDEO_DECODER:
        print("Building with CPU video decoder support")

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
                # This is an awful name. It should be "cpu_video_decoder". Keeping for BC.
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

    if USE_GPU_VIDEO_DECODER:
        # Locating GPU video decoder headers and libraries
        # CUDA_HOME should be set to the cuda root directory.
        # TORCHVISION_INCLUDE and TORCHVISION_LIBRARY should include the locations
        # to the headers and libraries below
        if not (
            BUILD_CUDA_SOURCES
            and CUDA_HOME is not None
            and any([os.path.exists(os.path.join(folder, "cuviddec.h")) for folder in TORCHVISION_INCLUDE])
            and any([os.path.exists(os.path.join(folder, "nvcuvid.h")) for folder in TORCHVISION_INCLUDE])
            and any([os.path.exists(os.path.join(folder, "libnvcuvid.so")) for folder in TORCHVISION_LIBRARY])
            and any([os.path.exists(os.path.join(folder, "libavcodec", "bsf.h")) for folder in ffmpeg_include_dir])
        ):
            print("Could not find necessary dependencies. Refer the setup.py to check which ones are needed.")
            print("Building without GPU video decoder support")
            return extensions
        print("Building torchvision with GPU video decoder support")

        gpu_decoder_path = os.path.join(CSRS_DIR, "io", "decoder", "gpu")
        gpu_decoder_src = glob.glob(os.path.join(gpu_decoder_path, "*.cpp"))
        cuda_libs = os.path.join(CUDA_HOME, "lib64")
        cuda_inc = os.path.join(CUDA_HOME, "include")

        _, extra_compile_args = get_macros_and_flags()
        extensions.append(
            CUDAExtension(
                "torchvision.gpu_decoder",
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

    print(f"Building wheel {package_name}-{version}")

    with open("README.md") as f:
        readme = f.read()

    extensions = [
        make_C_extension(),
        make_image_extension(),
        *make_video_decoders_extensions(),
    ]

    setup(
        name=package_name,
        version=version,
        author="PyTorch Core Team",
        author_email="soumith@pytorch.org",
        url="https://github.com/pytorch/vision",
        description="image and video datasets and models for torch deep learning",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD",
        packages=find_packages(exclude=("test",)),
        package_data={package_name: ["*.dll", "*.dylib", "*.so", "prototype/datasets/_builtin/*.categories"]},
        zip_safe=False,
        install_requires=get_requirements(),
        extras_require={
            "gdown": ["gdown>=4.7.3"],
            "scipy": ["scipy"],
        },
        ext_modules=extensions,
        python_requires=">=3.9",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
