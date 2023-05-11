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

PACKAGING_DIR="${PWD}/packaging"

if [[ $OS_TYPE == macos ]]; then
  JOBS=$(sysctl -n hw.logicalcpu)
else
  JOBS=$(nproc)
fi

if [[ $OS_TYPE == windows ]]; then
  echo $CUDA_PATH
  echo $CUDA_HOME
  echo $PATH
  export PATH="${CUDA_PATH}/libnvvp:${PATH}"
  JOBS=$(sysctl -n hw.logicalcpu)
fi


Torch_DIR=$(python -c "import pathlib, torch; print(pathlib.Path(torch.__path__[0]).joinpath('share/cmake/Torch'))")
if [[ "${GPU_ARCH_TYPE}" == "cuda" ]]; then
  WITH_CUDA=1
else
  WITH_CUDA=0
fi

echo '::group::Prepare CMake builds'
mkdir -p cpp_build

pushd test/tracing/frcnn
python trace_model.py
mkdir -p build
mv fasterrcnn_resnet50_fpn.pt build
popd

pushd examples/cpp/hello_world
python trace_model.py
mkdir -p build
mv resnet18.pt build
popd

# This was only needed for the tracing above
pip uninstall -y torchvision
echo '::endgroup::'

echo '::group::Build and install libtorchvision'
pushd cpp_build

cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}" -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}"
if [[ $OS_TYPE == windows ]]; then
  "${PACKAGING_DIR}/windows/internal/vc_env_helper.bat" "${PACKAGING_DIR}/windows/internal/build_cmake.bat" $JOBS
else
  make -j$JOBS
  make install
fi

popd
echo '::endgroup::'

echo '::group::Build and run project that uses Faster-RCNN'
pushd test/tracing/frcnn/build

cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}"
if [[ $OS_TYPE == windows ]]; then
  "${PACKAGING_DIR}/windows/internal/vc_env_helper.bat" "${PACKAGING_DIR}/windows/internal/build_frcnn.bat" $JOBS
else
  make -j$JOBS
fi

./test_frcnn_tracing

popd
echo '::endgroup::'

echo '::group::Build and run C++ example'
pushd examples/cpp/hello_world/build

cmake .. -DTorch_DIR="${Torch_DIR}"
if [[ $OS_TYPE == windows ]]; then
  "${PACKAGING_DIR}/windows/internal/vc_env_helper.bat" "${PACKAGING_DIR}/windows/internal/build_cpp_example.bat" $JOBS
else
  make -j$JOBS
fi

./hello-world

popd
echo '::endgroup::'
