#!/usr/bin/env bash

set -euxo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

####

Torch_DIR=$(python -c "import torch; print(torch.__path__[0])")/share/cmake/Torch
if [[ "${GPU_ARCH_TYPE}" == "cuda" ]]; then
  WITH_CUDA=1
else
  WITH_CUDA=0
fi

PARALLELISM=8
set +u
if [ -n "$MAX_JOBS" ]; then
    PARALLELISM=$MAX_JOBS
fi
set -u

####

mkdir cpp_build
pushd cpp_build

cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}"

make -j$PARALLELISM
make install

popd

####

python setup.py develop

####

pushd test/tracing/frcnn
mkdir build

python trace_model.py
cp fasterrcnn_resnet50_fpn.pt build

cd build
cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}"
make -j$PARALLELISM

./test_frcnn_tracing

popd

####

pushd examples/cpp/hello_world
mkdir build

python trace_model.py
cp resnet18.pt build

cd build
cmake .. -DTorch_DIR="${Torch_DIR}"

make -j$PARALLELISM

./hello-world
