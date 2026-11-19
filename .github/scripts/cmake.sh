#!/usr/bin/env bash

set -euxo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
set +x && eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci && set -x

# Setup the OS_TYPE environment variable that should be used for conditions involving the OS below.
case $(uname) in
  Linux)
    OS_TYPE=linux
    ;;
  Darwin)
    OS_TYPE=macos
    ;;
  MSYS*)
    OS_TYPE=windows
    ;;
  *)
    echo "Unknown OS type:" $(uname)
    exit 1
    ;;
esac

if [[ $OS_TYPE == macos ]]; then
  JOBS=$(sysctl -n hw.logicalcpu)
else
  JOBS=$(nproc)
fi

if [[ $OS_TYPE == linux ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

TORCH_PATH=$(python -c "import pathlib, torch; print(pathlib.Path(torch.__path__[0]))")
if [[ $OS_TYPE == windows ]]; then
  PACKAGING_DIR="${PWD}/packaging"
  export PATH="${TORCH_PATH}/lib:${PATH}"
fi

Torch_DIR="${TORCH_PATH}/share/cmake/Torch"
if [[ "${GPU_ARCH_TYPE}" == "cuda" ]]; then
  WITH_CUDA=1
else
  WITH_CUDA=0
fi

echo '::group::Build and install libtorchvision'
mkdir -p cpp_build
pushd cpp_build


# On macOS, CMake is looking for the library (*.dylib) and the header (*.h) separately. By default, it prefers to load
# the header from other packages that install the library. This easily leads to a mismatch if the library installed
# from conda doesn't have the exact same version. Thus, we need to explicitly set CMAKE_FIND_FRAMEWORK=NEVER to force
# it to not load anything from other installed frameworks. Resources:
# https://stackoverflow.com/questions/36523911/osx-homebrew-cmake-libpng-version-mismatch-issue
# https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_FRAMEWORK.html
cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}" \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DCMAKE_FIND_FRAMEWORK=NEVER \
  -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}"
if [[ $OS_TYPE == windows ]]; then
  "${PACKAGING_DIR}/windows/internal/vc_env_helper.bat" "${PACKAGING_DIR}/windows/internal/build_cmake.bat" $JOBS
else
  make -j$JOBS
  make install
fi

popd
echo '::endgroup::'
