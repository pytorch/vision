#!/usr/bin/env bash

set -euxo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

Torch_DIR=$(python -c "import torch; print(torch.__path__[0])")/share/cmake/Torch
if [[ "${GPU_ARCH_TYPE}" == "cuda" ]]; then
  WITH_CUDA=1
else
  WITH_CUDA=0
fi

mkdir cpp_build
cd cpp_build

cmake .. -DTorch_DIR="${Torch_DIR}" -DWITH_CUDA="${WITH_CUDA}"

make -j$PARALLELISM
make install
