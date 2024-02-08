import distutils.command.clean
import distutils.spawn
import glob
import os
import shutil
import subprocess
import sys

import torch
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension


def read(*names, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


cwd = os.path.dirname(os.path.abspath(__file__))

version_txt = os.path.join(cwd, "version.txt")
with open(version_txt) as f:
    version = f.readline().strip()
sha = "Unknown"
package_name = "torchvision"

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


def write_version_file():
    version_path = os.path.join(cwd, "torchvision", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write("from torchvision.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


pytorch_dep = "torch"
if os.getenv("PYTORCH_VERSION"):
    pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")

requirements = [
    "numpy",
    pytorch_dep,
]

# Excluding 8.3.* because of https://github.com/pytorch/vision/issues/4934
pillow_ver = " >= 5.3.0, !=8.3.*"
pillow_req = "pillow-simd" if get_dist("pillow-simd") is not None else "pillow"
requirements.append(pillow_req + pillow_ver)


def find_library(name, vision_include):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_prefix = os.environ.get("BUILD_PREFIX", None)
    is_conda_build = build_prefix is not None

    library_found = False
    conda_installed = False
    lib_folder = None
    include_folder = None
    library_header = f"{name}.h"

    # Lookup in TORCHVISION_INCLUDE or in the package file
    package_path = [os.path.join(this_dir, "torchvision")]
    for folder in vision_include + package_path:
        candidate_path = os.path.join(folder, library_header)
        library_found = os.path.exists(candidate_path)
        if library_found:
            break

    if not library_found:
        print(f"Running build on conda-build: {is_conda_build}")
        if is_conda_build:
            # Add conda headers/libraries
            if os.name == "nt":
                build_prefix = os.path.join(build_prefix, "Library")
            include_folder = os.path.join(build_prefix, "include")
            lib_folder = os.path.join(build_prefix, "lib")
            library_header_path = os.path.join(include_folder, library_header)
            library_found = os.path.isfile(library_header_path)
            conda_installed = library_found
        else:
            # Check if using Anaconda to produce wheels
            conda = shutil.which("conda")
            is_conda = conda is not None
            print(f"Running build on conda: {is_conda}")
            if is_conda:
                python_executable = sys.executable
                py_folder = os.path.dirname(python_executable)
                if os.name == "nt":
                    env_path = os.path.join(py_folder, "Library")
                else:
                    env_path = os.path.dirname(py_folder)
                lib_folder = os.path.join(env_path, "lib")
                include_folder = os.path.join(env_path, "include")
                library_header_path = os.path.join(include_folder, library_header)
                library_found = os.path.isfile(library_header_path)
                conda_installed = library_found

        if not library_found:
            if sys.platform == "linux":
                library_found = os.path.exists(f"/usr/include/{library_header}")
                library_found = library_found or os.path.exists(f"/usr/local/include/{library_header}")

    return library_found, conda_installed, include_folder, lib_folder


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchvision", "csrc")

    main_file = (
        glob.glob(os.path.join(extensions_dir, "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "autocast", "*.cpp"))
    )
    source_cpu = (
        glob.glob(os.path.join(extensions_dir, "ops", "autograd", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "cpu", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "quantized", "cpu", "*.cpp"))
    )
    source_mps = glob.glob(os.path.join(extensions_dir, "ops", "mps", "*.mm"))

    print("Compiling extensions with following flags:")
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    print(f"  FORCE_CUDA: {force_cuda}")
    force_mps = os.getenv("FORCE_MPS", "0") == "1"
    print(f"  FORCE_MPS: {force_mps}")
    debug_mode = os.getenv("DEBUG", "0") == "1"
    print(f"  DEBUG: {debug_mode}")
    use_png = os.getenv("TORCHVISION_USE_PNG", "1") == "1"
    print(f"  TORCHVISION_USE_PNG: {use_png}")
    use_jpeg = os.getenv("TORCHVISION_USE_JPEG", "1") == "1"
    print(f"  TORCHVISION_USE_JPEG: {use_jpeg}")
    use_nvjpeg = os.getenv("TORCHVISION_USE_NVJPEG", "1") == "1"
    print(f"  TORCHVISION_USE_NVJPEG: {use_nvjpeg}")
    use_ffmpeg = os.getenv("TORCHVISION_USE_FFMPEG", "1") == "1"
    print(f"  TORCHVISION_USE_FFMPEG: {use_ffmpeg}")
    use_video_codec = os.getenv("TORCHVISION_USE_VIDEO_CODEC", "1") == "1"
    print(f"  TORCHVISION_USE_VIDEO_CODEC: {use_video_codec}")

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    print(f"  NVCC_FLAGS: {nvcc_flags}")

    is_rocm_pytorch = False

    if torch.__version__ >= "1.5":
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (torch.version.hip is not None) and (ROCM_HOME is not None)

    if is_rocm_pytorch:
        from torch.utils.hipify import hipify_python

        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="torchvision/csrc/ops/cuda/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )
        source_cuda = glob.glob(os.path.join(extensions_dir, "ops", "hip", "*.hip"))
        # Copy over additional files
        for file in glob.glob(r"torchvision/csrc/ops/cuda/*.h"):
            shutil.copy(file, "torchvision/csrc/ops/hip")
    else:
        source_cuda = glob.glob(os.path.join(extensions_dir, "ops", "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": []}
    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            if nvcc_flags == "":
                nvcc_flags = []
            else:
                nvcc_flags = nvcc_flags.split(" ")
        else:
            define_macros += [("WITH_HIP", None)]
            nvcc_flags = []
        extra_compile_args["nvcc"] = nvcc_flags
    elif torch.backends.mps.is_available() or force_mps:
        sources += source_mps

    if sys.platform == "win32":
        define_macros += [("torchvision_EXPORTS", None)]
        define_macros += [("USE_PYTHON", None)]
        extra_compile_args["cxx"].append("/MP")

    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [f for f in nvcc_flags if not ("-O" in f or "-g" in f)]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")
    else:
        print("Compiling with debug mode OFF")
        extra_compile_args["cxx"].append("-g0")

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "torchvision._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    # ------------------- Torchvision extra extensions ------------------------
    vision_include = os.environ.get("TORCHVISION_INCLUDE", None)
    vision_library = os.environ.get("TORCHVISION_LIBRARY", None)
    vision_include = vision_include.split(os.pathsep) if vision_include is not None else []
    vision_library = vision_library.split(os.pathsep) if vision_library is not None else []
    include_dirs += vision_include
    library_dirs = vision_library

    # Image reading extension
    image_macros = []
    image_include = [extensions_dir]
    image_library = []
    image_link_flags = []

    if sys.platform == "win32":
        image_macros += [("USE_PYTHON", None)]

    # Locating libPNG
    libpng = shutil.which("libpng-config")
    pngfix = shutil.which("pngfix")
    png_found = libpng is not None or pngfix is not None

    use_png = use_png and png_found
    if use_png:
        print("Found PNG library")
        if libpng is not None:
            # Linux / Mac
            min_version = "1.6.0"
            png_version = subprocess.run([libpng, "--version"], stdout=subprocess.PIPE)
            png_version = png_version.stdout.strip().decode("utf-8")
            png_version = parse_version(png_version)
            if png_version >= parse_version(min_version):
                print("Building torchvision with PNG image support")
                png_lib = subprocess.run([libpng, "--libdir"], stdout=subprocess.PIPE)
                png_lib = png_lib.stdout.strip().decode("utf-8")
                if "disabled" not in png_lib:
                    image_library += [png_lib]
                png_include = subprocess.run([libpng, "--I_opts"], stdout=subprocess.PIPE)
                png_include = png_include.stdout.strip().decode("utf-8")
                _, png_include = png_include.split("-I")
                image_include += [png_include]
                image_link_flags.append("png")
                print(f"  libpng version: {png_version}")
                print(f"  libpng include path: {png_include}")
            else:
                print("Could not add PNG image support to torchvision:")
                print(f"  libpng minimum version {min_version}, found {png_version}")
                use_png = False
        else:
            # Windows
            png_lib = os.path.join(os.path.dirname(os.path.dirname(pngfix)), "lib")
            png_include = os.path.join(os.path.dirname(os.path.dirname(pngfix)), "include", "libpng16")
            image_library += [png_lib]
            image_include += [png_include]
            image_link_flags.append("libpng")
    else:
        print("Building torchvision without PNG image support")
    image_macros += [("PNG_FOUND", str(int(use_png)))]

    # Locating libjpeg
    (jpeg_found, jpeg_conda, jpeg_include, jpeg_lib) = find_library("jpeglib", vision_include)

    use_jpeg = use_jpeg and jpeg_found
    if use_jpeg:
        print("Building torchvision with JPEG image support")
        print(f"  libjpeg include path: {jpeg_include}")
        print(f"  libjpeg lib path: {jpeg_lib}")
        image_link_flags.append("jpeg")
        if jpeg_conda:
            image_library += [jpeg_lib]
            image_include += [jpeg_include]
    else:
        print("Building torchvision without JPEG image support")
    image_macros += [("JPEG_FOUND", str(int(use_jpeg)))]

    # Locating nvjpeg
    # Should be included in CUDA_HOME for CUDA >= 10.1, which is the minimum version we have in the CI
    nvjpeg_found = (
        extension is CUDAExtension
        and CUDA_HOME is not None
        and os.path.exists(os.path.join(CUDA_HOME, "include", "nvjpeg.h"))
    )

    use_nvjpeg = use_nvjpeg and nvjpeg_found
    if use_nvjpeg:
        print("Building torchvision with NVJPEG image support")
        image_link_flags.append("nvjpeg")
    else:
        print("Building torchvision without NVJPEG image support")
    image_macros += [("NVJPEG_FOUND", str(int(use_nvjpeg)))]

    image_path = os.path.join(extensions_dir, "io", "image")
    image_src = glob.glob(os.path.join(image_path, "*.cpp")) + glob.glob(os.path.join(image_path, "cpu", "*.cpp"))

    if is_rocm_pytorch:
        image_src += glob.glob(os.path.join(image_path, "hip", "*.cpp"))
        # we need to exclude this in favor of the hipified source
        image_src.remove(os.path.join(image_path, "image.cpp"))
    else:
        image_src += glob.glob(os.path.join(image_path, "cuda", "*.cpp"))

    if use_png or use_jpeg:
        ext_modules.append(
            extension(
                "torchvision.image",
                image_src,
                include_dirs=image_include + include_dirs + [image_path],
                library_dirs=image_library + library_dirs,
                define_macros=image_macros,
                libraries=image_link_flags,
                extra_compile_args=extra_compile_args,
            )
        )

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

    use_ffmpeg = use_ffmpeg and has_ffmpeg

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
            for search_path in ffmpeg_include_dir + include_dirs:
                full_path = os.path.join(search_path, library, "*.h")
                library_found |= len(glob.glob(full_path)) > 0

            if not library_found:
                print("Building torchvision without ffmpeg support")
                print(f"  {library} header files were not found, disabling ffmpeg support")
                use_ffmpeg = False
    else:
        print("Building torchvision without ffmpeg support")

    if use_ffmpeg:
        print("Building torchvision with ffmpeg support")
        print(f"  ffmpeg version: {ffmpeg_version}")
        print(f"  ffmpeg include path: {ffmpeg_include_dir}")
        print(f"  ffmpeg library_dir: {ffmpeg_library_dir}")

        # TorchVision base decoder + video reader
        video_reader_src_dir = os.path.join(this_dir, "torchvision", "csrc", "io", "video_reader")
        video_reader_src = glob.glob(os.path.join(video_reader_src_dir, "*.cpp"))
        base_decoder_src_dir = os.path.join(this_dir, "torchvision", "csrc", "io", "decoder")
        base_decoder_src = glob.glob(os.path.join(base_decoder_src_dir, "*.cpp"))
        # Torchvision video API
        videoapi_src_dir = os.path.join(this_dir, "torchvision", "csrc", "io", "video")
        videoapi_src = glob.glob(os.path.join(videoapi_src_dir, "*.cpp"))
        # exclude tests
        base_decoder_src = [x for x in base_decoder_src if "_test.cpp" not in x]

        combined_src = video_reader_src + base_decoder_src + videoapi_src

        ext_modules.append(
            CppExtension(
                "torchvision.video_reader",
                combined_src,
                include_dirs=[
                    base_decoder_src_dir,
                    video_reader_src_dir,
                    videoapi_src_dir,
                    extensions_dir,
                    *ffmpeg_include_dir,
                    *include_dirs,
                ],
                library_dirs=ffmpeg_library_dir + library_dirs,
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
        extension is CUDAExtension
        and CUDA_HOME is not None
        and any([os.path.exists(os.path.join(folder, "cuviddec.h")) for folder in vision_include])
        and any([os.path.exists(os.path.join(folder, "nvcuvid.h")) for folder in vision_include])
        and any([os.path.exists(os.path.join(folder, "libnvcuvid.so")) for folder in library_dirs])
    )

    use_video_codec = use_video_codec and video_codec_found
    if (
        use_video_codec
        and use_ffmpeg
        and any([os.path.exists(os.path.join(folder, "libavcodec", "bsf.h")) for folder in ffmpeg_include_dir])
    ):
        print("Building torchvision with video codec support")
        gpu_decoder_path = os.path.join(extensions_dir, "io", "decoder", "gpu")
        gpu_decoder_src = glob.glob(os.path.join(gpu_decoder_path, "*.cpp"))
        cuda_libs = os.path.join(CUDA_HOME, "lib64")
        cuda_inc = os.path.join(CUDA_HOME, "include")

        ext_modules.append(
            extension(
                "torchvision.Decoder",
                gpu_decoder_src,
                include_dirs=include_dirs + [gpu_decoder_path] + [cuda_inc] + ffmpeg_include_dir,
                library_dirs=ffmpeg_library_dir + library_dirs + [cuda_libs],
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

    return ext_modules


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
    print(f"Building wheel {package_name}-{version}")

    write_version_file()

    with open("README.md") as f:
        readme = f.read()

    setup(
        # Metadata
        name=package_name,
        version=version,
        author="PyTorch Core Team",
        author_email="soumith@pytorch.org",
        url="https://github.com/pytorch/vision",
        description="image and video datasets and models for torch deep learning",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test",)),
        package_data={package_name: ["*.dll", "*.dylib", "*.so"]},
        zip_safe=False,
        install_requires=requirements,
        extras_require={
            "scipy": ["scipy"],
        },
        ext_modules=get_extensions(),
        python_requires=">=3.8",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
