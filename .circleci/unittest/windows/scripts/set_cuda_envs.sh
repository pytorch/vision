#!/usr/bin/env bash

if [[ -z "${CUDA_VERSION}" || "${CUDA_VERSION}" == "cpu" ]] ; then
    exit 0
fi

version=$CUDA_VERSION

# set cuda envs
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/libnvvp:$PATH"
export CUDA_PATH_V${version/./_}="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
export CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
export NVTOOLSEXT_PATH="C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64"
export FORCE_CUDA=1

if  [ ! -d "$CUDA_PATH" ]
then
    echo "$CUDA_PATH" does not exist
    exit 1
fi

if [ ! -f "${CUDA_PATH}\include\nvjpeg.h" ]
then
    echo "nvjpeg does not exist"
    exit 1
fi

# check cuda driver version
for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
    if [[ -x "$path" ]]; then
        "$path" || echo "true";
        break
    fi
done
which nvcc
nvcc --version
env | grep CUDA
