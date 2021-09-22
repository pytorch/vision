#!/usr/bin/env bash
set -ex

echo CU_VERSION is "${CU_VERSION}"
echo CUDA_VERSION is "${CUDA_VERSION}"

# Currenly, CU_VERSION and CUDA_VERSION are not consistent. 
# to understand this code, please checck out https://github.com/pytorch/vision/issues/4443
version="cpu"
if [[ ! -z "${CUDA_VERSION}" ]] ; then
    version="$CUDA_VERSION"
else
    if [[ ${#CU_VERSION} -eq 5 ]]; then
        version="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
fi

if [[ "$version" != "cpu" ]]; then
    exit /b 0
fi

# set cuda envs
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/libnvvp:$PATH"
export CUDA_PATH_V${version/./_}="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"

if  [ ! -d "$CUDA_PATH" ]
then
    echo "$CUDA_PATH" does not exist
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
