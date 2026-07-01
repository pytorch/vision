import importlib.machinery
import os

from torch.hub import _get_torch_home


_HOME = os.path.join(_get_torch_home(), "datasets", "vision")
_USE_SHARDED_DATASETS = False
IN_FBCODE = False


def _download_file_from_remote_location(fpath: str, url: str) -> None:
    pass


def _is_remote_location_available() -> bool:
    return False


try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


def _preload_image_stable_cuda_libs():
    # image_stable links nvjpeg, which torch does not bundle, so on a wheel install without a
    # system CUDA it may not be on the loader path and image_stable fails to load. We try to
    # locate nvjpeg in the CUDA toolkit (CUDA_PATH/CUDA_HOME, then nvcc) and preload it; on
    # Windows a .pyd import does not search PATH, so we also scan PATH, where the CUDA
    # installer's DLL dir lands. No-op if already loaded or not found.
    import ctypes
    import glob
    import shutil

    win = os.name == "nt"
    subdirs = ("bin", os.path.join("bin", "x64")) if win else ("lib64", "lib")
    pattern = "nvjpeg64_*.dll" if win else "libnvjpeg.so*"

    def _load(path):
        return ctypes.WinDLL(path) if win else ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    cuda_homes = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    if nvcc := shutil.which("nvcc"):
        cuda_homes.append(os.path.dirname(os.path.dirname(nvcc)))
    lib_dirs = []
    for cuda_home in filter(None, cuda_homes):
        lib_dirs += [os.path.join(cuda_home, d) for d in subdirs]
        lib_dirs += glob.glob(os.path.join(cuda_home, "targets", "*", "lib"))
    if win:
        lib_dirs += os.environ.get("PATH", "").split(os.pathsep)
    for lib_dir in lib_dirs:
        for path in sorted(glob.glob(os.path.join(lib_dir, pattern)), reverse=True):
            try:
                _load(path)
                return
            except OSError:
                continue


def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)
    if os.name == "nt":
        # Register the main torchvision library location on the default DLL path
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        os.add_dll_directory(lib_dir)

        kernel32.SetErrorMode(prev_error_mode)
    if lib_name == "image_stable":
        _preload_image_stable_cuda_libs()

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"Could not find module '{lib_name}' in {lib_dir}")

    return ext_specs.origin
